FROM rust:1.94-bookworm AS build
ARG TARGETARCH
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl xz-utils python3 libunwind-dev && rm -rf /var/lib/apt/lists/*
RUN case "$TARGETARCH" in \
      arm64) arch=aarch64; checksum=ea4b09bfb22ec6f6c6ceac57ab63efb6b46e17ab08d21f69f3a48b38e1534f17 ;; \
      amd64) arch=x86_64; checksum=70e49664a74374b48b51e6f3fdfbf437f6395d42509050588bd49abe52ba3d00 ;; \
      *) exit 1 ;; \
    esac && \
    curl -fsSL "https://ziglang.org/download/0.16.0/zig-${arch}-linux-0.16.0.tar.xz" -o /tmp/zig.tar.xz && \
    echo "$checksum  /tmp/zig.tar.xz" | sha256sum -c - && \
    mkdir /opt/zig && tar -xJf /tmp/zig.tar.xz -C /opt/zig --strip-components=1 && rm /tmp/zig.tar.xz
ENV PATH="/opt/zig:${PATH}"
COPY source/ /workspace/
WORKDIR /workspace/affon
RUN zig build -Doptimize=ReleaseFast --prefix /out

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends libunwind8 && rm -rf /var/lib/apt/lists/*
COPY --from=build /out/bin/affon /usr/local/bin/affon
COPY --from=build /workspace/affon/packages /opt/affon/packages
COPY --from=build /workspace/affon/apps/vision-serving/src /opt/affon/apps/vision-serving/src
WORKDIR /opt/affon
ENV RUNTIME_PACKAGE_PATH=/opt/affon/packages MODEL_DIR=/mnt/models MODEL_NAME=vision HOST=0.0.0.0 PORT=8080 AFFON_DEVICE=cpu
USER 65532:65532
EXPOSE 8080
ENTRYPOINT ["affon", "apps/vision-serving/src/server.ts"]
