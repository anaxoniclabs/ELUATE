#!/bin/bash
#
# Eluate Installer
#
# Flags:
#   --no-rc-edit   Don't append a PATH line to ~/.zshrc or ~/.bashrc.
#                  Use this if you manage your shell config yourself —
#                  you'll need to add $HOME/.eluate to PATH manually.
#

set -e

NO_RC_EDIT=0
for arg in "$@"; do
    case "$arg" in
        --no-rc-edit) NO_RC_EDIT=1 ;;
        -h|--help)
            echo "Usage: $0 [--no-rc-edit]"
            echo ""
            echo "  --no-rc-edit   Skip editing ~/.zshrc or ~/.bashrc to add"
            echo "                 \$HOME/.eluate to PATH."
            exit 0
            ;;
    esac
done

# Orange color palette
ORANGE='\033[38;2;251;126;58m'
ORANGE_LIGHT='\033[38;2;252;166;109m'
ORANGE_DARK='\033[38;2;196;94;32m'
ORANGE_MUTED='\033[38;2;166;124;82m'
NC='\033[0m'

echo -e "${ORANGE}"
echo "██╗  ██╗██╗   ██╗███████╗██╗  ██╗███████╗██████╗ "
echo "██║  ██║██║   ██║██╔════╝██║  ██║██╔════╝██╔══██╗"
echo "███████║██║   ██║███████╗███████║█████╗  ██████╔╝"
echo "██╔══██║██║   ██║╚════██║██╔══██║██╔══╝  ██╔══██╗"
echo "██║  ██║╚██████╔╝███████║██║  ██║███████╗██║  ██║"
echo "╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝"
echo -e "${NC}"
echo -e "${ORANGE_MUTED}Remove background music from any video${NC}"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${ORANGE_DARK}Python 3.10+ required. Please install Python first.${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    echo -e "${ORANGE_DARK}Python 3.10+ required (found $PYTHON_VERSION)${NC}"
    exit 1
fi
echo -e "${ORANGE_MUTED}Python $PYTHON_VERSION${NC}"

# Check FFmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo -e "${ORANGE}Installing FFmpeg...${NC}"
    if command -v brew &> /dev/null; then
        brew install ffmpeg
    else
        echo -e "${ORANGE_DARK}FFmpeg required. Install with: brew install ffmpeg${NC}"
        exit 1
    fi
else
    echo -e "${ORANGE_MUTED}FFmpeg installed${NC}"
fi

# Setup
ELUATE_DIR="$HOME/.eluate"
mkdir -p "$ELUATE_DIR/models"

echo ""
echo -e "${ORANGE}Setting up environment...${NC}"
python3 -m venv "$ELUATE_DIR/venv"
source "$ELUATE_DIR/venv/bin/activate"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Eluate's separator imports from vendor/mss-training (a git submodule).
# If we're installing from a git checkout, initialize the submodule here so
# the first `eluate video.mp4` invocation doesn't abort on "Vendor submodule
# missing". A tarball/pip install has no .git and skips this.
if [ -d "$PROJECT_DIR/.git" ] && [ -f "$PROJECT_DIR/.gitmodules" ]; then
    echo -e "${ORANGE}Initializing vendor submodule...${NC}"
    (cd "$PROJECT_DIR" && git submodule update --init --recursive)
fi

if [ -f "$PROJECT_DIR/pyproject.toml" ]; then
    pip install --upgrade pip -q
    pip install -e "$PROJECT_DIR" -q
else
    pip install --upgrade pip -q
    pip install eluate -q
fi

# Download model
# Keep this digest in sync with CHECKPOINT_SHA256[("bandit-v2", "multi")]
# in eluate/utils/paths.py.
MODEL_URL="https://zenodo.org/records/12701995/files/checkpoint-multi.ckpt?download=1"
MODEL_PATH="$ELUATE_DIR/models/checkpoint-multi.ckpt"
MODEL_PART="$MODEL_PATH.part"
EXPECTED_SHA="abcfccf65446752a057f4a302c941479a54b7560ebf8d7bca039d2ea98e64cfc"

verify_checkpoint_sha() {
    local path="$1"

    if [ "$(uname)" == "Darwin" ]; then
        FILE_SIZE=$(stat -f%z "$path" 2>/dev/null || echo "0")
    else
        FILE_SIZE=$(stat -c%s "$path" 2>/dev/null || echo "0")
    fi

    if [ "$FILE_SIZE" -lt 400000000 ]; then
        echo -e "${ORANGE_DARK}Checkpoint is too small; refusing to use it.${NC}"
        return 1
    fi

    if command -v shasum &> /dev/null; then
        ACTUAL_SHA=$(shasum -a 256 "$path" | awk '{print $1}')
    elif command -v sha256sum &> /dev/null; then
        ACTUAL_SHA=$(sha256sum "$path" | awk '{print $1}')
    else
        echo -e "${ORANGE_DARK}Neither shasum nor sha256sum available; cannot verify checkpoint.${NC}"
        return 1
    fi

    if [ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]; then
        echo -e "${ORANGE_DARK}Checkpoint SHA-256 mismatch — refusing to continue.${NC}"
        echo -e "${ORANGE_DARK}  expected: $EXPECTED_SHA${NC}"
        echo -e "${ORANGE_DARK}  actual:   $ACTUAL_SHA${NC}"
        return 1
    fi
}

if [ -f "$MODEL_PATH" ]; then
    if verify_checkpoint_sha "$MODEL_PATH"; then
        echo -e "${ORANGE_MUTED}Model already downloaded and verified${NC}"
    else
        echo -e "${ORANGE_DARK}Delete $MODEL_PATH and rerun the installer to download a fresh copy.${NC}"
        exit 1
    fi
else
    echo ""
    echo -e "${ORANGE}Downloading AI model (450 MB)...${NC}"
    rm -f "$MODEL_PART"
    curl -L --progress-bar "$MODEL_URL" -o "$MODEL_PART"

    if ! verify_checkpoint_sha "$MODEL_PART"; then
        rm -f "$MODEL_PART"
        exit 1
    fi
    mv "$MODEL_PART" "$MODEL_PATH"
    echo -e "${ORANGE_MUTED}Checkpoint SHA-256 verified${NC}"
fi

# Copy bandit-v2 config alongside the checkpoint. Source lives inside
# the package (eluate/configs/) so the same path works for editable
# installs from a clone and pip installs from a wheel.
if [ -f "$PROJECT_DIR/eluate/configs/bandit_v2.yaml" ]; then
    cp "$PROJECT_DIR/eluate/configs/bandit_v2.yaml" "$ELUATE_DIR/models/config_bandit_v2.yaml"
fi

# Create launcher
cat > "$ELUATE_DIR/eluate" << 'LAUNCHER'
#!/bin/bash
source "$HOME/.eluate/venv/bin/activate"
python -m eluate "$@"
LAUNCHER
chmod +x "$ELUATE_DIR/eluate"

# Add to PATH
SHELL_RC=""
if [[ "$SHELL" == *"zsh"* ]]; then
    SHELL_RC="$HOME/.zshrc"
elif [[ "$SHELL" == *"bash"* ]]; then
    SHELL_RC="$HOME/.bashrc"
fi

if [[ "$NO_RC_EDIT" -eq 1 ]]; then
    echo ""
    echo -e "${ORANGE_MUTED}Skipping shell rc edit (--no-rc-edit).${NC}"
    echo -e "${ORANGE_MUTED}Add this line to your shell config manually:${NC}"
    echo '    export PATH="$HOME/.eluate:$PATH"'
elif [[ -n "$SHELL_RC" ]]; then
    if ! grep -q '\.eluate' "$SHELL_RC" 2>/dev/null; then
        echo 'export PATH="$HOME/.eluate:$PATH"' >> "$SHELL_RC"
    fi
fi

echo ""
echo -e "${ORANGE_LIGHT}Installation complete!${NC}"
echo ""
echo -e "${ORANGE}Usage:${NC}"
echo "  eluate              Interactive mode"
echo "  eluate video.mp4    Process a video"
echo ""
echo -e "${ORANGE_MUTED}Open a new terminal to start using Eluate${NC}"
