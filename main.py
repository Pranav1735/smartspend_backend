"""
SMART SPENDER - PRODUCTION FASTAPI (Render Ready)
"""

# =============================================
# IMPORTS
# =============================================
import os, cv2, re, joblib, json, hashlib, datetime as dt, numpy as np, pandas as pd, traceback, logging
from datetime import datetime
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, DateTime, func, ForeignKey
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy import exc as sa_exc

logging.getLogger().setLevel(logging.INFO)

# =============================================
# DATABASE SETUP
# =============================================
DATABASE_URL = "sqlite:///./smart_spender.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Receipt(Base):
    __tablename__ = "receipts"
    id = Column(Integer, primary_key=True)
    vendor = Column(String(255))
    date = Column(Date)
    total = Column(Float)
    receipt_hash = Column(String(64), unique=True, index=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    items = relationship("ReceiptItem", back_populates="receipt", cascade="all, delete-orphan")

class ReceiptItem(Base):
    __tablename__ = "receipt_items"
    id = Column(Integer, primary_key=True)
    receipt_id = Column(Integer, ForeignKey("receipts.id"))
    name = Column(String(512))
    price = Column(Float)
    total = Column(Float)
    category = Column(String(255))
    confidence = Column(Float)
    receipt = relationship("Receipt", back_populates="items")

Base.metadata.create_all(bind=engine)
print("✅ Database initialized!")

# =============================================
# PYDANTIC MODELS
# =============================================
class ReceiptItemModel(BaseModel):
    name: str
    price: float
    total: float

class ManualReceipt(BaseModel):
    vendor: str
    date: str
    total: float
    items: List[ReceiptItemModel]

# =============================================
# OCR SERVICE
# =============================================
import easyocr
class OCRService:
    def __init__(self):
        print("📷 Initializing OCR...")
        self.reader = easyocr.Reader(["en"], gpu=False)
        print("✅ OCR Ready!")

    def process_image_ocr(self, image_bytes: bytes) -> dict:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (0, 0), fx=1.3, fy=1.3)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 31, 2)
        results = self.reader.readtext(img, detail=1)
        text = " ".join([r[1] for r in results])

        date_match = re.search(r"(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})", text)
        total_match = re.search(r"(?:Total|Amount)[^\d]*(\d+\.\d{0,2})", text)
        vendor_match = re.search(r"(?i)(store|mart|shop|restaurant|cafe)\s?[A-Za-z]*", text)

        items = []
        for line in text.split("\n"):
            price_match = re.search(r"(\d+\.\d{0,2})\s*$", line)
            if price_match:
                try:
                    price = float(price_match.group(1))
                except:
                    continue
                name = line[:price_match.start()].strip()
                if len(name) > 2:
                    items.append({"name": name, "price": price, "total": price})

        return {
            "vendor": vendor_match.group(0) if vendor_match else "Unknown Store",
            "date": date_match.group(1) if date_match else datetime.now().strftime("%Y-%m-%d"),
            "total": float(total_match.group(1)) if total_match else sum(i["total"] for i in items),
            "items": items
        }

# =============================================
# CATEGORIZATION SERVICE
# =============================================
class CategorizationService:
    def __init__(self, confidence_threshold: float = 0.60):
        self.ml_loaded = False
        self.confidence_threshold = confidence_threshold
        self.load_models()

    def load_models(self):
        print("🤖 Loading Categorization Models...")
        os.makedirs("models", exist_ok=True)
        try:
            if os.path.exists("models/category_model.pkl"):
                self.ml_model = joblib.load("models/category_model.pkl")
                self.vectorizer = joblib.load("models/vectorizer.pkl")
                self.scaler = joblib.load("models/scaler.pkl")
                self.ml_loaded = True
                print("✅ ML Categorization loaded!")
            else:
                print("⚠️ Models not found. Using rule-based categorization.")
        except Exception:
            traceback.print_exc()
            self.ml_loaded = False

    def _fallback_categorize(self, item_name: str) -> str:
        item_lower = item_name.lower()
        keywords = {
            "Food": ["food","pizza","burger","juice","coffee","meal","biryani","dosa"],
            "Groceries": ["rice","milk","bread","vegetable","fruit","oil"],
            "Fashion": ["shirt","jeans","shoes","dress","bag","jacket"],
            "Electronics": ["charger","phone","laptop","earphone","tv","fridge"],
            "Healthcare": ["medicine","tablet","hospital","syrup","pill"],
            "Fuel": ["petrol","diesel","fuel","cng"],
            "Transportation": ["cab","ride","bus","ticket","auto","metro"],
            "Utilities": ["electricity","water","gas","bill","recharge"]
        }
        for category, words in keywords.items():
            if any(word in item_lower for word in words):
                return category
        return "Uncategorized"

    def _preprocess(self, s: str) -> str:
        s = s.lower().strip()
        s = re.sub(r'[^a-z0-9\s]', ' ', s)
        s = re.sub(r'\s+', ' ', s)
        return s

    def predict_category(self, item_name: str, amount: float) -> tuple:
        txt = self._preprocess(item_name)
        if not self.ml_loaded:
            return self._fallback_categorize(item_name), 0.6

        try:
            text_features = self.vectorizer.transform([txt]).toarray()
            numeric_features = self.scaler.transform(
                [[amount, datetime.now().day, datetime.now().month, datetime.now().year]]
            )
            X = np.hstack([numeric_features, text_features])
            probs = self.ml_model.predict_proba(X)[0]
            idx = int(np.argmax(probs))
            category = self.ml_model.classes_[idx]
            confidence = float(probs[idx])

            if confidence < self.confidence_threshold:
                return self._fallback_categorize(item_name), confidence
            return category, confidence
        except Exception:
            logging.exception("Exception during predict_category")
            return self._fallback_categorize(item_name), 0.5

# =============================================
# FORECAST + DATABASE SERVICES
# =============================================
class ForecastService:
    def generate_forecasts(self):
        session = SessionLocal()
        try:
            receipts = session.query(Receipt).all()
            if len(receipts) < 3:
                return {"message": "Need at least 3 receipts for forecasting"}
            data = []
            for r in receipts:
                for it in r.items:
                    data.append({"category": it.category, "amount": it.total, "date": r.date})
            df = pd.DataFrame(data)
            result = []
            for c in df["category"].unique():
                cat_df = df[df["category"] == c]
                cur = cat_df["amount"].sum()
                if len(cat_df) >= 3:
                    recent = cat_df.tail(3)["amount"].mean()
                    older = cat_df.head(3)["amount"].mean()
                    growth = (recent - older) / older if older > 0 else 0.1
                    forecast = cur * (1 + growth)
                else:
                    forecast = cur
                result.append({
                    "category": c,
                    "current": round(cur,2),
                    "forecast": round(forecast,2),
                    "trend": "rising" if forecast > cur else "stable"
                })
            total_cur = sum(r["current"] for r in result)
            total_fore = sum(r["forecast"] for r in result)
            return {
                "next_month_forecast": round(total_fore,2),
                "current_month_total": round(total_cur,2),
                "category_forecasts": result,
                "growth_percentage": round(((total_fore - total_cur)/total_cur*100),2) if total_cur>0 else 0
            }
        finally:
            session.close()

class DatabaseService:
    def compute_hash(self,vendor,total,date):
        return hashlib.sha256(f"{vendor}{total}{date}".encode()).hexdigest()

    def save_receipt(self,data,cat_items):
        s = SessionLocal()
        try:
            rhash = self.compute_hash(data["vendor"],data["total"],data["date"])
            existing = s.query(Receipt).filter_by(receipt_hash=rhash).first()
            if existing:
                return {"status":"duplicate","receipt_id":existing.id}
            try:
                date_obj = datetime.strptime(data["date"],"%Y-%m-%d").date()
            except:
                date_obj = datetime.now().date()
            r = Receipt(vendor=data["vendor"], date=date_obj,
                        total=data["total"], receipt_hash=rhash)
            s.add(r); s.flush()
            for i,c in zip(data["items"],cat_items):
                s.add(ReceiptItem(receipt_id=r.id, name=i["name"], price=i["price"],
                                  total=i["total"], category=c["category"], confidence=c["confidence"]))
            try:
                s.commit()
            except sa_exc.IntegrityError:
                s.rollback()
                return {"status":"duplicate","receipt_id":existing.id if existing else None}
            return {"status":"created","receipt_id":r.id}
        finally:
            s.close()

    def get_analytics(self):
        s = SessionLocal()
        try:
            receipts = s.query(Receipt).all()
            total = sum(r.total for r in receipts) if receipts else 0
            cats = s.query(ReceiptItem.category, func.sum(ReceiptItem.total)).group_by(ReceiptItem.category).all()
            data = [{"category":c,"total":float(t),"percentage":round(t/total*100,2) if total>0 else 0}
                    for c,t in cats]
            return {"total_spent":round(total,2),"total_receipts":len(receipts),"category_breakdown":data}
        finally:
            s.close()

# =============================================
# PIPELINE
# =============================================
class SmartSpenderPipeline:
    def __init__(self):
        self.ocr = OCRService()
        self.cat = CategorizationService(confidence_threshold=0.60)
        self.forecast = ForecastService()
        self.db = DatabaseService()
        print("🚀 PIPELINE READY!")

    async def process_ocr(self,img_bytes):
        data = self.ocr.process_image_ocr(img_bytes)
        categorized_items = []
        for i in data["items"]:
            c, conf = self.cat.predict_category(i["name"], i["total"])
            categorized_items.append({"name": i["name"], "price": i["price"], "category": c, "confidence": conf})
        db_result = self.db.save_receipt(data, [{"category": it["category"], "confidence": it["confidence"]} for it in categorized_items])
        return {
            "status":"success",
            "receipt_id": db_result.get("receipt_id"),
            "duplicate": db_result.get("status") == "duplicate",
            "vendor": data.get("vendor"),
            "total": data.get("total"),
            "items": categorized_items,
            "analytics": self.db.get_analytics(),
            "forecasts": self.forecast.generate_forecasts()
        }

    async def process_manual(self,data):
        categorized_items = []
        for i in data["items"]:
            c, conf = self.cat.predict_category(i["name"], i["total"])
            categorized_items.append({"name": i["name"], "price": i["price"], "category": c, "confidence": conf})
        db_result = self.db.save_receipt(data, [{"category": it["category"], "confidence": it["confidence"]} for it in categorized_items])
        return {
            "status":"success",
            "receipt_id": db_result.get("receipt_id"),
            "duplicate": db_result.get("status") == "duplicate",
            "vendor": data.get("vendor"),
            "total": data.get("total"),
            "items": categorized_items,
            "analytics": self.db.get_analytics(),
            "forecasts": self.forecast.generate_forecasts()
        }

# =============================================
# FASTAPI APP
# =============================================
app = FastAPI(title="Smart Spender API")

@app.get("/")
def root_redirect():
    return RedirectResponse(url="/docs")

@app.get("/api")
def api_root_info():
    return JSONResponse({
        "message": "Welcome to Smart Spender API 🚀",
        "docs_url": "/docs",
        "health_check": "/api/health"
    })

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
pipeline = SmartSpenderPipeline()

@app.post("/api/upload-receipt")
async def upload(file: UploadFile = File(...)):
    return await pipeline.process_ocr(await file.read())

@app.post("/api/manual-receipt")
async def manual_debug(receipt: ManualReceipt):
    try:
        d = {"vendor":receipt.vendor,"date":receipt.date,"total":receipt.total,
             "items":[i.dict() for i in receipt.items]}
        return await pipeline.process_manual(d)
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"{str(e)}\n\n{tb}")

@app.get("/api/analytics")
async def analytics(): 
    return pipeline.db.get_analytics()

@app.get("/api/forecasts")
async def forecasts(): 
    return pipeline.forecast.generate_forecasts()

@app.get("/api/receipts")
async def get_receipts():
    session = SessionLocal()
    try:
        receipts = session.query(Receipt).all()
        return [{"id": r.id, "vendor": r.vendor, "date": str(r.date), "total": r.total} for r in receipts]
    finally:
        session.close()

@app.get("/api/health")
async def health(): 
    return {"status":"healthy","categorization":pipeline.cat.ml_loaded}

@app.get("/api/version")
def version():
    return {
        "model_loaded": pipeline.cat.ml_loaded,
        "classes": list(getattr(pipeline.cat.ml_model, "classes_", [])) if pipeline.cat.ml_loaded else []
    }
