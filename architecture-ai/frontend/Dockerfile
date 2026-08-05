# Dockerfile for the static HTML/CSS/JS frontend.
# No build step needed (plain HTML/CSS/JS) — just serve the files.
FROM nginx:1.27-alpine

# Remove Nginx's default sample site
RUN rm -rf /usr/share/nginx/html/*

# Copy the whole frontend project (pages/, nav/, assets/, index.html, etc.)
# as-is into Nginx's web root, preserving folder structure so relative
# paths like ../nav/nav.js keep working exactly as they do today.
COPY . /usr/share/nginx/html/

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD wget -q --spider http://localhost:80/ || exit 1