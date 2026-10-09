# qli-Client (qubic.li official pool) for macOS Apple Silicon.
#
# IMPORTANT: the **x64** tarball (glibc Linux) is used and the image runs
# linux/amd64 under Rosetta in the colima VM, because the pool pushes an
# Android (bionic) worker to Linux arm64 clients — upstream bug
# github.com/qubic-li/client/issues/136. See docker-compose.yml.
#
# The tarball is downloaded once into qli-cache/ (see README) so the build is
# hermetic and reproducible. Expected SHA256:
#   3.9.1 x64:   127101cdf726cc0116bb981c78260b8583b377fec0ded54acafebd6174f1c340
#   3.9.1 arm64: 1b314d6996efded6e307e96919049c35fa746a00576af335165d47f892a11072

FROM debian:bookworm-slim

ARG QLI_VERSION=3.9.1
ARG QLI_ARCH=x64

# ca-certificates: WSS to the pool | libicu: .NET runtime | tzdata: local-time logs
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates libicu72 tzdata \
 && rm -rf /var/lib/apt/lists/*

COPY qli-cache/qli-Client-${QLI_VERSION}-Linux-${QLI_ARCH}.tar.gz /tmp/qli.tar.gz

RUN mkdir -p /qli \
 && tar -xzf /tmp/qli.tar.gz -C /qli \
 && rm -f /tmp/qli.tar.gz \
 && chmod +x /qli/qli-Client /qli/qli-Service.sh \
 && useradd --system --uid 1000 --home /qli qli \
 && chown -R qli:qli /qli

WORKDIR /qli
USER qli

# Configuration is 100% via environment variables (ClientSettings__*), .NET style.
# See docker-compose.yml. Never mount your seed here: public address/token only.

CMD ["/qli/qli-Client"]
