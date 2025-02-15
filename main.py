from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
from crews import crews  # Import the crew definitions

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to "*" if testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class CrewRequest(BaseModel):
    crew_name: str
    inputs: Dict[str, Any]

@app.post("/execute_crew/")
async def execute_crew(request: CrewRequest):
    # Check if the crew exists
    if request.crew_name not in crews:
        raise HTTPException(status_code=404, detail="Crew not found")

    crew_definition = crews[request.crew_name]

    # Validate required inputs
    missing_keys = [
        key for key in crew_definition.required_inputs.keys() if key not in request.inputs
    ]
    if missing_keys:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required inputs: {', '.join(missing_keys)}",
        )
     # Filter inputs: include only valid required and optional keys
    valid_inputs = {k: v for k, v in request.inputs.items() if k in crew_definition.required_inputs or k in crew_definition.optional_inputs}

    # Execute the selected crew
    output = crew_definition.crew.kickoff(inputs=valid_inputs)

    return {"crew_name": request.crew_name, "output": output}
