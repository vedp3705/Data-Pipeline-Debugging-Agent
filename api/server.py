from fastapi import FastAPI
from agent.agent import diagnose
import uvicorn

app = FastAPI()

@app.get("/diagnose")
def get_diagnosis():
    result = diagnose()
    return {"diagnosis": result}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
