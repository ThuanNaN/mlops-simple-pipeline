# Machine Learning Operations (MLOps) Simple Pipeline

## 2. Backend API
- FastAPI - https://fastapi.tiangolo.com
- Unicorn - https://www.uvicorn.org

## 3. Frontend Interface
- Gradio - https://www.gradio.app

## 4. Expose local server

### 4.1 Require
- Ngrok - https://ngrok.com

### 4.2 Configuration
#### Ubuntu/Linux
```bash 
# Check config path
ngrok config check
```

```bash 
# Login https://ngrok.com to take auth token
# config.yml

version: "2"
authtoken: <token>
region: ap
tunnels:
    webapp:
        addr: 3000
        proto: http
    api:
        addr: 5000
        proto: http
```

### 4.3 Start server
```bash
ngrok start --all --config config.yml
```
