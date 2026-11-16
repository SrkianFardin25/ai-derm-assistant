from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import Response, RedirectResponse

from src.api.routes import router
from src.core.config import get_settings
from src.core.lifespan import lifespan
from src.core.exceptions import register_exception_handlers

settings = get_settings()

# Lock documentation endpoints in production unless explicitly enabled via ENABLE_DOCS
is_docs_enabled = settings.ENABLE_DOCS or (settings.ENVIRONMENT != "production")
docs_url = "/docs" if is_docs_enabled else None
redoc_url = "/redoc" if is_docs_enabled else None
openapi_url = "/openapi.json" if is_docs_enabled else None

app = FastAPI(
    title="DermAssist AI — Clinical Inference API",
    description="Production skin lesion diagnostic support, ONNX inference, occlusion-sensitivity heatmaps, and Groq LPU clinical narrative advisory.",
    version="2.0.1",
    docs_url=docs_url,
    redoc_url=redoc_url,
    openapi_url=openapi_url,
    lifespan=lifespan,
)

# Cross-Origin Resource Sharing (CORS)
# When wildcard '*' is configured, allow_credentials MUST be False to adhere to W3C security standard
is_wildcard = "*" in settings.ALLOWED_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=not is_wildcard,
    allow_methods=["GET", "POST", "HEAD", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def security_headers_and_https_middleware(request: Request, call_next):
    """Enforce HTTPS redirection and attach enterprise-grade HTTP security headers."""
    # 1. Enforce HTTPS behind reverse proxy (Render / Cloudflare / Vercel)
    if (
        settings.FORCE_HTTPS
        and settings.ENVIRONMENT == "production"
        and request.headers.get("x-forwarded-proto") == "http"
    ):
        url = request.url.replace(scheme="https")
        return RedirectResponse(url=str(url), status_code=status.HTTP_301_MOVED_PERMANENTLY)

    response: Response = await call_next(request)

    # 2. Strict Transport Security (HSTS - 2 years, subdomains, preload)
    response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains; preload"
    # 3. Prevent MIME Sniffing
    response.headers["X-Content-Type-Options"] = "nosniff"
    # 4. Anti-Clickjacking Frame Protection
    response.headers["X-Frame-Options"] = "DENY"
    # 5. Referrer Policy
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    # 6. Content Security Policy
    response.headers["Content-Security-Policy"] = "default-src 'self'; frame-ancestors 'none';"
    # 7. Permissions Policy
    response.headers["Permissions-Policy"] = "camera=(self), geolocation=(self), microphone=()"

    return response


app.include_router(router)
register_exception_handlers(app)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
