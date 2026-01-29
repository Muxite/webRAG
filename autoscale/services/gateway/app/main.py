import os
import uvicorn
from app.api import app

if __name__ == "__main__":
    forwarded_allow_ips = os.environ.get("FORWARDED_ALLOW_IPS", "*")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8080,
        proxy_headers=True,
        forwarded_allow_ips=forwarded_allow_ips
    )

