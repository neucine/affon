FROM rust:1.94-bookworm AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates cmake curl libcurl4-openssl-dev libffi-dev \
      libssl-dev libunwind-dev libzmq3-dev patch pkg-config python3 xz-utils \
    && rm -rf /var/lib/apt/lists/*
COPY cyna/ /opt/cyna/
ENV PATH="/opt/cyna/bin:${PATH}"
COPY source/ /workspace/
WORKDIR /workspace/hao
RUN bash ./build_deps.sh && sh ./tools/prepare_std_modules.sh
WORKDIR /workspace/affon
RUN mkdir -p /out/bin && cyna build . --output /out/bin/affon

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
      libcurl4 libffi8 libssl3 libunwind8 libzmq5 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /out/bin/affon /usr/local/bin/affon
COPY --from=build /workspace/affon/packages /opt/affon/packages
COPY --from=build /workspace/affon/apps/vision-serving/src /opt/affon/apps/vision-serving/src
WORKDIR /opt/affon
ENV RUNTIME_PACKAGE_PATH=/opt/affon/packages MODEL_DIR=/mnt/models MODEL_NAME=vision HOST=0.0.0.0 PORT=8080 AFFON_DEVICE=cpu
USER 65532:65532
EXPOSE 8080
ENTRYPOINT ["affon", "apps/vision-serving/src/server.ts"]
