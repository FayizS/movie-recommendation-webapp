from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import httpx
import sqlite3
import joblib
import pandas as pd


# DB Setup
conn = sqlite3.connect("clicks.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS clicks (
  movie_id INTEGER,
  rating REAL,
  popularity REAL,
  genre INTEGER,
  clicked INTEGER
)
""")
conn.commit()

model = joblib.load("model.pkl")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/static/assets", StaticFiles(directory="static/assets"), name="assets")  
app.mount("/static/vendor", StaticFiles(directory="static/vendor"), name="vendor")

templates = Jinja2Templates(directory="templates")

TMDB_API_KEY = "your key here"


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"https://api.themoviedb.org/3/movie/popular?api_key={TMDB_API_KEY}&language=en-US&page=1"
        )
        data = response.json()
        movies = data.get("results", [])

    recommended = []

    for m in movies:
        X_pred = pd.DataFrame([{
                                "rating": m["vote_average"]#,
                                #"genre": m["genre_ids"][0] if m["genre_ids"] else 0
                            }])
        prob = model.predict_proba(X_pred)[0][1]

        m["score"] = prob
        recommended.append(m)

    recommended = sorted(recommended, key=lambda x: x["score"], reverse=True)[:8]

    
    for m in movies:
        c.execute("INSERT INTO clicks VALUES (?,?,?,?,?)",
              (m["id"], m["vote_average"], m["popularity"],
               m["genre_ids"][0] if m["genre_ids"] else 0, 0))
    conn.commit()

    

    return templates.TemplateResponse("index.html",{"request" : request, "movies":movies, "recommended":recommended})

@app.post("/click")
async def click(
    movie_id: int = Form(...),
    rating: float = Form(...),
    popularity: float = Form(...),
    genre: int = Form(...)
):
    c.execute("INSERT INTO clicks VALUES (?,?,?,?,?)",
              (movie_id, rating, popularity, genre, 1))
    conn.commit()
    return RedirectResponse("/", status_code=302)

if __name__ == "__main__":
    uvicorn.run("main:app", host = "127.0.0.1", port=8000, reload=True)
       