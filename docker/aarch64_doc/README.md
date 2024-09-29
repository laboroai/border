Run the following commands to create docs in border/doc.

## Create image

```bash
# In border/docker/aarch64_headless
sh build.sh
```

## Run container

```bash
# in border/docker/aarch64_doc
sh run.sh
```

## Create docs in the above container

```bash
# in border/docker/aarch64_doc
sh doc.sh
```
