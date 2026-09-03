FROM nimlang/nim:latest

WORKDIR /app

COPY . /app

RUN nimble install -y
RUN rm -f nimble.paths && \
    nim c -d:release -d:ssl --opt:speed --mm:orc --threads:off -o:bin/lunatic src/lunatic.nim && \
    nim c -d:release -d:ssl --opt:speed --mm:orc --threads:off -o:bin/server src/server.nim

EXPOSE 18080

CMD ["./bin/lunatic", "serve", "--port", "18080", "--db", "lunatic_cognitive.db"]
