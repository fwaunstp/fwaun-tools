#!/usr/bin/env bash
# fwaun-tools installer for macOS and Linux.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/fwaunstp/fwaun-tools/main/install.sh | sh
#   curl -fsSL https://raw.githubusercontent.com/fwaunstp/fwaun-tools/main/install.sh | sh -s -- --both
#   curl -fsSL https://raw.githubusercontent.com/fwaunstp/fwaun-tools/main/install.sh | sh -s -- --cli-only
#
# Installs FULL builds of the CLI (fwaun-tools) and GUI (fwaun-tools-gui).
# On a headless Linux host (no $DISPLAY / $WAYLAND_DISPLAY) only the CLI is
# installed; override the auto-detect with --both / --cli-only / --gui-only.
#
# The prebuilt binaries link ONNX Runtime and need glibc 2.38+ on Linux. On an
# older distro (Debian 12, Ubuntu 22.04, …) build the portable light CLI from
# source instead (no ONNX Runtime, no glibc floor):
#   cargo install --git https://github.com/fwaunstp/fwaun-tools fwaun-tools-cli
#
# Flags:
#   --version <tag>   release tag to install (default: latest)
#   --prefix <dir>    install root for binaries (default: $HOME/.local)
#   --both            install CLI + GUI (override headless auto-detect)
#   --cli-only        install CLI only
#   --gui-only        install GUI only
#   --no-verify       skip SHA256 verification

set -euo pipefail

REPO="fwaunstp/fwaun-tools"
VERSION="latest"
PREFIX="${HOME}/.local"
MODE=auto            # auto | both | cli | gui — resolved after OS detection
VERIFY=1

err() { printf 'error: %s\n' "$*" >&2; exit 1; }
info() { printf '==> %s\n' "$*"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --version)    VERSION="$2"; shift 2 ;;
        --prefix)     PREFIX="$2"; shift 2 ;;
        --both)       MODE=both; shift ;;
        --cli-only)   MODE=cli; shift ;;
        --gui-only)   MODE=gui; shift ;;
        --no-verify)  VERIFY=0; shift ;;
        -h|--help)
            sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'
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
    err "Intel macOS prebuilt binaries are not published. Build from source with: cargo install --git https://github.com/${REPO} fwaun-tools-cli"
fi

TARGET="${OS}-${ARCH}"
info "platform: ${TARGET}"

# Resolve which binaries to install. `auto` (the default) installs the GUI too
# unless we're on a headless Linux host — the GUI needs a display at runtime,
# so there's no point dropping it on a server. macOS is always desktop-capable.
case "$MODE" in
    both) INSTALL_CLI=1; INSTALL_GUI=1 ;;
    cli)  INSTALL_CLI=1; INSTALL_GUI=0 ;;
    gui)  INSTALL_CLI=0; INSTALL_GUI=1 ;;
    auto)
        INSTALL_CLI=1
        if [ "$OS" = "macos" ] || [ -n "${DISPLAY:-}" ] || [ -n "${WAYLAND_DISPLAY:-}" ]; then
            INSTALL_GUI=1
        else
            INSTALL_GUI=0
            info "headless Linux (no \$DISPLAY/\$WAYLAND_DISPLAY): installing CLI only — pass --both to include the GUI"
        fi
        ;;
esac

# Prebuilt binaries link ONNX Runtime, which needs glibc 2.38+. Best-effort
# warn on older systems and point at the source-built light CLI.
if [ "$OS" = "linux" ] && command -v ldd >/dev/null 2>&1; then
    GLIBC_VER="$(ldd --version 2>/dev/null | head -n1 | grep -oE '[0-9]+\.[0-9]+' | head -n1 || true)"
    if [ -n "$GLIBC_VER" ]; then
        GLIBC_MAJOR="${GLIBC_VER%%.*}"
        GLIBC_MINOR="${GLIBC_VER#*.}"
        if [ "$GLIBC_MAJOR" = "2" ] && [ "$GLIBC_MINOR" -lt 38 ] 2>/dev/null; then
            info "warning: system glibc ${GLIBC_VER} is older than 2.38 — the prebuilt binaries link ONNX Runtime and will fail to start."
            info "         build the portable light CLI from source instead (no ONNX, runs anywhere):"
            info "           cargo install --git https://github.com/${REPO} fwaun-tools-cli"
        fi
    fi
fi

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

# CLI install — single binary tarball with fwaun-tools at the root.
if [ "$INSTALL_CLI" = "1" ]; then
    CLI_NAME="fwaun-tools-cli-${VER}-${TARGET}.tar.gz"
    download "$CLI_NAME"
    mkdir -p "${PREFIX}/bin"
    tar xzf "$TMPDIR/${CLI_NAME}" -C "${PREFIX}/bin"
    chmod +x "${PREFIX}/bin/fwaun-tools"
    info "installed CLI: ${PREFIX}/bin/fwaun-tools"
fi

# GUI install — the GUI archive is a tar.gz of a directory containing
# both binaries. We extract just the fwaun-tools-gui binary alongside
# the CLI in $PREFIX/bin (uniform across macOS and Linux now that the
# macOS .app wrapper has been dropped).
if [ "$INSTALL_GUI" = "1" ]; then
    GUI_NAME="fwaun-tools-${VER}-${TARGET}.tar.gz"
    download "$GUI_NAME"
    EXTRACT_DIR="$TMPDIR/extract"
    mkdir -p "$EXTRACT_DIR" "${PREFIX}/bin"
    tar xzf "$TMPDIR/${GUI_NAME}" -C "$EXTRACT_DIR"
    INNER_DIR="$(find "$EXTRACT_DIR" -mindepth 1 -maxdepth 1 -type d | head -n1)"
    cp "$INNER_DIR/fwaun-tools-gui" "${PREFIX}/bin/fwaun-tools-gui"
    chmod +x "${PREFIX}/bin/fwaun-tools-gui"
    if [ "$OS" = "macos" ]; then
        # Strip the quarantine bit so Gatekeeper doesn't block first launch.
        xattr -d com.apple.quarantine "${PREFIX}/bin/fwaun-tools-gui" 2>/dev/null || true
    fi
    info "installed GUI: ${PREFIX}/bin/fwaun-tools-gui"
fi

case ":${PATH}:" in
    *":${PREFIX}/bin:"*) ;;
    *) printf '\nnote: %s/bin is not on $PATH. Add this to your shell rc:\n  export PATH="%s/bin:$PATH"\n' "$PREFIX" "$PREFIX" ;;
esac

info "done."
