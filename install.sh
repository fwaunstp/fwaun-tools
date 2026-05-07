#!/usr/bin/env bash
# anima-tagger installer for macOS and Linux.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/fwaunstp/anima-tagger/main/install.sh | sh
#   curl -fsSL https://raw.githubusercontent.com/fwaunstp/anima-tagger/main/install.sh | sh -s -- --version v0.2.1
#   curl -fsSL https://raw.githubusercontent.com/fwaunstp/anima-tagger/main/install.sh | sh -s -- --cli-only
#
# Flags:
#   --version <tag>   release tag to install (default: latest)
#   --prefix <dir>    install root for binaries (default: $HOME/.local)
#   --cli-only        skip GUI install
#   --gui-only        skip CLI install
#   --no-verify       skip SHA256 verification

set -euo pipefail

REPO="fwaunstp/anima-tagger"
VERSION="latest"
PREFIX="${HOME}/.local"
INSTALL_CLI=1
INSTALL_GUI=1
VERIFY=1

err() { printf 'error: %s\n' "$*" >&2; exit 1; }
info() { printf '==> %s\n' "$*"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --version)    VERSION="$2"; shift 2 ;;
        --prefix)     PREFIX="$2"; shift 2 ;;
        --cli-only)   INSTALL_GUI=0; shift ;;
        --gui-only)   INSTALL_CLI=0; shift ;;
        --no-verify)  VERIFY=0; shift ;;
        -h|--help)
            sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) err "unknown flag: $1" ;;
    esac
done

command -v curl >/dev/null 2>&1 || err "curl is required"
command -v tar  >/dev/null 2>&1 || err "tar is required"

OS_RAW="$(uname -s)"
ARCH_RAW="$(uname -m)"
case "$OS_RAW" in
    Darwin) OS=macos ;;
    Linux)  OS=linux ;;
    *) err "unsupported OS: $OS_RAW" ;;
esac
case "$ARCH_RAW" in
    arm64|aarch64) ARCH=arm64 ;;
    x86_64|amd64)  ARCH=x64 ;;
    *) err "unsupported arch: $ARCH_RAW" ;;
esac

if [ "$OS" = "macos" ] && [ "$ARCH" = "x64" ]; then
    err "Intel macOS prebuilt binaries are not published. Build from source with: cargo install --git https://github.com/${REPO} anima-tagger-cli"
fi

TARGET="${OS}-${ARCH}"
info "platform: ${TARGET}"

if [ "$VERSION" = "latest" ]; then
    info "resolving latest release"
    TAG="$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" \
        | grep -m1 '"tag_name":' \
        | sed -E 's/.*"tag_name": *"([^"]+)".*/\1/')"
    [ -n "$TAG" ] || err "could not resolve latest tag"
else
    TAG="$VERSION"
fi
VER="${TAG#v}"
info "version: ${TAG}"

BASE_URL="https://github.com/${REPO}/releases/download/${TAG}"
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

if [ "$VERIFY" = "1" ] && command -v sha256sum >/dev/null 2>&1; then
    info "downloading SHA256SUMS"
    curl -fsSL -o "$TMPDIR/SHA256SUMS" "${BASE_URL}/SHA256SUMS" || {
        info "SHA256SUMS not found on this release; skipping verification"
        VERIFY=0
    }
fi

verify() {
    [ "$VERIFY" = "1" ] || return 0
    [ -f "$TMPDIR/SHA256SUMS" ] || return 0
    ( cd "$TMPDIR" && grep -F " $1" SHA256SUMS | sha256sum -c - >/dev/null )
}

download() {
    local name="$1"
    info "downloading ${name}"
    curl -fL --retry 3 -o "$TMPDIR/${name}" "${BASE_URL}/${name}"
    verify "$name" || err "checksum verification failed for ${name}"
}

# CLI install — single binary tarball with anima-tagger at the root.
if [ "$INSTALL_CLI" = "1" ]; then
    CLI_NAME="anima-tagger-cli-${VER}-${TARGET}.tar.gz"
    download "$CLI_NAME"
    mkdir -p "${PREFIX}/bin"
    tar xzf "$TMPDIR/${CLI_NAME}" -C "${PREFIX}/bin"
    chmod +x "${PREFIX}/bin/anima-tagger"
    info "installed CLI: ${PREFIX}/bin/anima-tagger"
fi

# GUI install — the GUI archive is a tar.gz of a directory containing
# both binaries. We extract just the anima-tagger-gui binary alongside
# the CLI in $PREFIX/bin (uniform across macOS and Linux now that the
# macOS .app wrapper has been dropped).
if [ "$INSTALL_GUI" = "1" ]; then
    GUI_NAME="anima-tagger-${VER}-${TARGET}.tar.gz"
    download "$GUI_NAME"
    EXTRACT_DIR="$TMPDIR/extract"
    mkdir -p "$EXTRACT_DIR" "${PREFIX}/bin"
    tar xzf "$TMPDIR/${GUI_NAME}" -C "$EXTRACT_DIR"
    INNER_DIR="$(find "$EXTRACT_DIR" -mindepth 1 -maxdepth 1 -type d | head -n1)"
    cp "$INNER_DIR/anima-tagger-gui" "${PREFIX}/bin/anima-tagger-gui"
    chmod +x "${PREFIX}/bin/anima-tagger-gui"
    if [ "$OS" = "macos" ]; then
        # Strip the quarantine bit so Gatekeeper doesn't block first launch.
        xattr -d com.apple.quarantine "${PREFIX}/bin/anima-tagger-gui" 2>/dev/null || true
    fi
    info "installed GUI: ${PREFIX}/bin/anima-tagger-gui"
fi

case ":${PATH}:" in
    *":${PREFIX}/bin:"*) ;;
    *) printf '\nnote: %s/bin is not on $PATH. Add this to your shell rc:\n  export PATH="%s/bin:$PATH"\n' "$PREFIX" "$PREFIX" ;;
esac

info "done."
