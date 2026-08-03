# Minimal allowlist proxy — see nginx.conf for why this exists. Runs non-root; all its
# writable paths (client_body_temp_path/proxy_temp_path/pid, all redirected to /tmp in
# nginx.conf) are covered by a --tmpfs /tmp at `docker run` time, so the image itself can
# stay --read-only just like every other codebench container.
FROM nginx:1.27-alpine

COPY nginx.conf /etc/nginx/nginx.conf

USER nginx
EXPOSE 8090
