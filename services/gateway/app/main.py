import os
import uvicorn
from app.api import app

if __name__ == "__main__":
    # Enable proxy headers when behind a load balancer (ALB)
    # This makes uvicorn trust X-Forwarded-* headers from the proxy
    trust_proxy = os.environ.get("TRUST_PROXY", "").lower() == "true"
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        proxy_headers=trust_proxy,
    )

