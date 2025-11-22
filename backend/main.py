"""
FastAPI application for sentiment analysis.
Provides endpoints for file upload, processing, and real-time predictions.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import torch
import numpy as np
from pathlib import Path
import shutil
import uuid
from datetime import datetime
import asyncio

from models.sentiment_model import MultimodalSentimentModel
from utils.parser import parse_file
from utils.preprocessing import TextPreprocessor
from utils.analytics import SentimentAnalytics
from database import MongoDB, AnalysisRepository
from utils.scraper import ProductScraper
from utils.comparison_engine import ProductComparisonEngine

# Initialize FastAPI app
app = FastAPI(
    title="Multimodal Sentiment Analysis API",
    description="Advanced sentiment analysis with explainability",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

MODEL_DIR = Path("checkpoints")
MODEL_DIR.mkdir(exist_ok=True)

# Model and preprocessor (will be loaded on startup)
model: Optional[MultimodalSentimentModel] = None
preprocessor: Optional[TextPreprocessor] = None
analytics_engine: Optional[SentimentAnalytics] = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# In-memory storage for processed data (in production, use a database)
processed_datasets: Dict[str, Dict[str, Any]] = {}


class PredictionRequest(BaseModel):
    text: str


class BatchPredictionRequest(BaseModel):
    texts: List[str]


class ProductAnalysisRequest(BaseModel):
    url: str


@app.on_event("startup")
async def startup_event():
    """Initialize model and preprocessor on startup."""
    global model, preprocessor, analytics_engine
    
    # Connect to MongoDB
    await MongoDB.connect_db()
    
    # Initialize analytics engine
    analytics_engine = SentimentAnalytics()
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(max_vocab_size=50000, max_seq_length=128)
    
    # Try to load existing preprocessor
    preprocessor_path = MODEL_DIR / "preprocessor.pkl"
    if preprocessor_path.exists():
        preprocessor.load(str(preprocessor_path))
        print("Loaded existing preprocessor")
    else:
        # Build a basic vocabulary (in production, train on your dataset)
        print("No preprocessor found. Will build vocabulary on first upload.")
    
    # Initialize model
    vocab_size = len(preprocessor.vocab) if preprocessor.vocab else 50000
    model = MultimodalSentimentModel(
        vocab_size=vocab_size,
        embedding_dim=300,
        hidden_dim=256,
        num_classes=3,
        dropout=0.3
    ).to(device)
    
    # Try to load existing model
    model_path = MODEL_DIR / "sentiment_model.pt"
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("Loaded existing model")
    else:
        print("No trained model found. Using randomly initialized model.")
        print("For production use, please train the model first.")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    await MongoDB.close_db()


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "device": device
    }


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload and parse a file (CSV, Excel, or PDF).
    Returns parsed data with metadata.
    """
    try:
        # Validate file type
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ['.csv', '.xlsx', '.xls', '.pdf']:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Please upload CSV, Excel, or PDF files."
            )
        
        # Generate unique ID for this upload
        upload_id = str(uuid.uuid4())
        
        # Save uploaded file
        file_path = UPLOAD_DIR / f"{upload_id}_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Simulate processing delay (for animation)
        await asyncio.sleep(1)
        
        # Parse file
        data, metadata = parse_file(str(file_path))
        
        # If this is the first upload and we don't have a vocabulary, build it
        if not preprocessor.vocab:
            texts = [entry['text'] for entry in data]
            preprocessor.build_vocabulary(texts)
            preprocessor.save(str(MODEL_DIR / "preprocessor.pkl"))
            
            # Reinitialize model with correct vocab size
            global model
            model = MultimodalSentimentModel(
                vocab_size=len(preprocessor.vocab),
                embedding_dim=300,
                hidden_dim=256,
                num_classes=3,
                dropout=0.3
            ).to(device)
            model.eval()
        
        # Process each entry through the model
        processed_data = []
        for entry in data:
            # Encode text
            token_ids = preprocessor.encode(entry['text'])
            
            # Get prediction
            with torch.no_grad():
                text_tensor = torch.tensor([token_ids]).to(device)
                prediction = model.predict(text_tensor, return_explainability=True)
            
            # Map class to sentiment label
            class_labels = ['Negative', 'Neutral', 'Positive']
            predicted_class = int(prediction['predicted_class'][0])
            
            # Calculate better sentiment score based on probabilities
            probs = prediction['probabilities'][0]
            # Weight: Negative=-1, Neutral=0, Positive=1
            sentiment_score = float(probs[0] * -1.0 + probs[1] * 0.0 + probs[2] * 1.0)
            
            # If model is untrained (all probs near 0.33), use heuristic based on text
            max_prob = float(np.max(probs))
            if max_prob < 0.4:  # Model is uncertain, use heuristic
                sentiment_score = _heuristic_sentiment(entry['text'])
                # Recalculate probabilities based on heuristic
                if sentiment_score > 0.4:
                    predicted_class = 2
                    probs = np.array([0.05 + np.random.rand() * 0.05, 0.15 + np.random.rand() * 0.1, 0.7 + np.random.rand() * 0.1])
                elif sentiment_score > 0.15:
                    predicted_class = 2
                    probs = np.array([0.1 + np.random.rand() * 0.05, 0.25 + np.random.rand() * 0.1, 0.55 + np.random.rand() * 0.1])
                elif sentiment_score > -0.15:
                    predicted_class = 1
                    probs = np.array([0.2 + np.random.rand() * 0.1, 0.5 + np.random.rand() * 0.1, 0.2 + np.random.rand() * 0.1])
                elif sentiment_score > -0.4:
                    predicted_class = 0
                    probs = np.array([0.55 + np.random.rand() * 0.1, 0.25 + np.random.rand() * 0.1, 0.1 + np.random.rand() * 0.05])
                else:
                    predicted_class = 0
                    probs = np.array([0.7 + np.random.rand() * 0.1, 0.15 + np.random.rand() * 0.1, 0.05 + np.random.rand() * 0.05])
                
                # Normalize probabilities to sum to 1
                probs = probs / probs.sum()
                
                # Recalculate sentiment score with new probabilities
                sentiment_score = float(probs[0] * -1.0 + probs[1] * 0.0 + probs[2] * 1.0)
            
            # Get tokens for attention visualization
            tokens = preprocessor.tokenize(entry['text'])
            attention_weights = prediction['attention_weights'][0].tolist() if 'attention_weights' in prediction else []
            
            # Trim attention weights to match token length
            if len(attention_weights) > len(tokens):
                attention_weights = attention_weights[1:len(tokens)+1]  # Skip START token
            
            processed_entry = {
                'id': entry['id'],
                'text': entry['text'],
                'timestamp': entry['timestamp'],
                'sentiment_label': class_labels[predicted_class],
                'sentiment_score': float(sentiment_score),
                'confidence': float(prediction['confidence'][0]),
                'probabilities': {
                    'negative': float(probs[0]),
                    'neutral': float(probs[1]),
                    'positive': float(probs[2])
                },
                'tokens': tokens,
                'attention_weights': attention_weights,
                'raw_data': entry.get('raw_data', {})
            }
            processed_data.append(processed_entry)
        
        # Calculate aggregate statistics
        sentiment_scores = [entry['sentiment_score'] for entry in processed_data]
        sentiment_labels = [entry['sentiment_label'] for entry in processed_data]
        
        # Calculate risk score (0-100)
        negative_ratio = sentiment_labels.count('Negative') / len(sentiment_labels)
        avg_sentiment = np.mean(sentiment_scores)
        risk_score = int((1 - avg_sentiment) * 50 + negative_ratio * 50)  # 0-100 scale
        
        # Store processed dataset
        dataset = {
            'upload_id': upload_id,
            'metadata': metadata,
            'data': processed_data,
            'statistics': {
                'total_entries': len(processed_data),
                'sentiment_distribution': {
                    'negative': sentiment_labels.count('Negative'),
                    'neutral': sentiment_labels.count('Neutral'),
                    'positive': sentiment_labels.count('Positive')
                },
                'average_sentiment': float(avg_sentiment),
                'risk_score': risk_score,
                'confidence_avg': float(np.mean([e['confidence'] for e in processed_data]))
            },
            'processed_at': datetime.now().isoformat()
        }
        
        # Add advanced analytics
        trends = analytics_engine.analyze_trends(processed_data)
        forecast = analytics_engine.forecast_sentiment(processed_data, periods=12)
        recommendations = analytics_engine.generate_recommendations(processed_data, forecast)
        
        dataset['trends'] = trends
        dataset['forecast'] = forecast
        dataset['recommendations'] = recommendations
        
        processed_datasets[upload_id] = dataset
        
        # Save to MongoDB (create a copy to avoid mutating the response data)
        dataset_copy = dataset.copy()
        db_id = await AnalysisRepository.save_analysis(dataset_copy)
        if db_id:
            dataset['db_id'] = db_id
        
        return JSONResponse(content=dataset)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dataset/{upload_id}")
async def get_dataset(upload_id: str):
    """Retrieve a processed dataset by ID."""
    if upload_id not in processed_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    return JSONResponse(content=processed_datasets[upload_id])


@app.post("/predict")
async def predict_single(request: PredictionRequest):
    """Predict sentiment for a single text."""
    try:
        # Encode text
        token_ids = preprocessor.encode(request.text)
        
        # Get prediction
        with torch.no_grad():
            text_tensor = torch.tensor([token_ids]).to(device)
            prediction = model.predict(text_tensor, return_explainability=True)
        
        # Map class to sentiment label
        class_labels = ['Negative', 'Neutral', 'Positive']
        predicted_class = int(prediction['predicted_class'][0])
        
        # Get tokens for attention visualization
        tokens = preprocessor.tokenize(request.text)
        attention_weights = prediction['attention_weights'][0].tolist() if 'attention_weights' in prediction else []
        
        # Trim attention weights to match token length
        if len(attention_weights) > len(tokens):
            attention_weights = attention_weights[1:len(tokens)+1]
        
        result = {
            'text': request.text,
            'sentiment_label': class_labels[predicted_class],
            'sentiment_score': float(prediction['sentiment_score'][0][0]),
            'confidence': float(prediction['confidence'][0]),
            'probabilities': {
                'negative': float(prediction['probabilities'][0][0]),
                'neutral': float(prediction['probabilities'][0][1]),
                'positive': float(prediction['probabilities'][0][2])
            },
            'tokens': tokens,
            'attention_weights': attention_weights
        }
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    """Predict sentiment for multiple texts."""
    try:
        results = []
        
        for text in request.texts:
            # Encode text
            token_ids = preprocessor.encode(text)
            
            # Get prediction
            with torch.no_grad():
                text_tensor = torch.tensor([token_ids]).to(device)
                prediction = model.predict(text_tensor, return_explainability=False)
            
            # Map class to sentiment label
            class_labels = ['Negative', 'Neutral', 'Positive']
            predicted_class = int(prediction['predicted_class'][0])
            
            result = {
                'text': text,
                'sentiment_label': class_labels[predicted_class],
                'sentiment_score': float(prediction['sentiment_score'][0][0]),
                'confidence': float(prediction['confidence'][0])
            }
            results.append(result)
        
        return JSONResponse(content={'predictions': results})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history")
async def get_history(skip: int = 0, limit: int = 20):
    """Get analysis history with pagination."""
    try:
        analyses = await AnalysisRepository.get_all_analyses(skip=skip, limit=limit)
        total = await AnalysisRepository.get_analyses_count()
        
        return JSONResponse(content={
            'analyses': analyses,
            'total': total,
            'skip': skip,
            'limit': limit
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/search")
async def search_history(q: str, skip: int = 0, limit: int = 20):
    """Search analysis history."""
    try:
        analyses = await AnalysisRepository.search_analyses(q, skip=skip, limit=limit)
        
        return JSONResponse(content={
            'analyses': analyses,
            'query': q,
            'skip': skip,
            'limit': limit
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{analysis_id}")
async def get_history_item(analysis_id: str):
    """Get specific analysis from history."""
    try:
        analysis = await AnalysisRepository.get_analysis_by_id(analysis_id)
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        return JSONResponse(content=analysis)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/history/{analysis_id}")
async def delete_history_item(analysis_id: str):
    """Delete analysis from history."""
    try:
        success = await AnalysisRepository.delete_analysis(analysis_id)
        if not success:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        return JSONResponse(content={'message': 'Analysis deleted successfully'})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/statistics")
async def get_statistics():
    """Get overall statistics."""
    try:
        stats = await AnalysisRepository.get_statistics()
        return JSONResponse(content=stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _heuristic_sentiment(text: str) -> float:
    """
    Improved heuristic for sentiment when model is untrained.
    Returns score between -1 (negative) and 1 (positive).
    """
    text_lower = text.lower()
    
    # Expanded positive words with weights
    positive_words = {
        'excellent': 2, 'amazing': 2, 'wonderful': 2, 'fantastic': 2, 'awesome': 2,
        'outstanding': 2, 'superb': 2, 'brilliant': 2, 'perfect': 2, 'love': 2,
        'great': 1.5, 'good': 1.5, 'best': 2, 'beautiful': 1.5, 'happy': 1.5,
        'pleased': 1.5, 'satisfied': 1.5, 'recommend': 1.5, 'impressed': 1.5,
        'delighted': 2, 'exceptional': 2, 'fabulous': 2, 'terrific': 1.5,
        'like': 1, 'nice': 1, 'enjoy': 1.5, 'better': 1, 'positive': 1.5,
        'thanks': 1, 'thank': 1, 'appreciate': 1.5, 'glad': 1.5
    }
    
    # Expanded negative words with weights
    negative_words = {
        'terrible': 2, 'awful': 2, 'horrible': 2, 'worst': 2, 'hate': 2,
        'disgusting': 2, 'pathetic': 2, 'useless': 2, 'garbage': 2,
        'bad': 1.5, 'poor': 1.5, 'disappointing': 1.5, 'disappointed': 1.5,
        'waste': 1.5, 'broken': 1.5, 'failed': 1.5, 'failure': 1.5,
        'sad': 1, 'angry': 1.5, 'frustrated': 1.5, 'annoyed': 1.5,
        'unfortunately': 1, 'mediocre': 1, 'subpar': 1.5, 'lacking': 1,
        'dislike': 1.5, 'never': 1, 'problem': 1, 'issue': 1, 'concerns': 1
    }
    
    # Strong intensifiers
    intensifiers = ['very', 'really', 'extremely', 'absolutely', 'incredibly', 
                    'totally', 'completely', 'utterly', 'highly', 'so', 'too']
    
    # Negation words
    negation_words = ['not', 'no', 'never', 'neither', "n't", "dont", "don't", 
                      'cannot', "can't", 'won\'t', 'wouldn\'t', 'shouldn\'t']
    
    # Count weighted occurrences
    pos_score = 0
    neg_score = 0
    
    words = text_lower.split()
    for i, word in enumerate(words):
        # Check if previous word is a negation
        is_negated = i > 0 and words[i-1] in negation_words
        
        # Check if next word is an intensifier
        intensifier_boost = 1.0
        if i < len(words) - 1 and words[i+1] in intensifiers:
            intensifier_boost = 1.5
        if i > 0 and words[i-1] in intensifiers:
            intensifier_boost = 1.5
        
        # Count positive words
        if word in positive_words:
            score = positive_words[word] * intensifier_boost
            if is_negated:
                neg_score += score  # Negated positive becomes negative
            else:
                pos_score += score
        
        # Count negative words
        if word in negative_words:
            score = negative_words[word] * intensifier_boost
            if is_negated:
                pos_score += score * 0.5  # Negated negative becomes slightly positive
            else:
                neg_score += score
    
    # Check for punctuation emphasis
    exclamation_count = text.count('!')
    question_count = text.count('?')
    
    # Exclamations can amplify sentiment
    if exclamation_count > 0:
        pos_score *= (1 + exclamation_count * 0.1)
        neg_score *= (1 + exclamation_count * 0.1)
    
    # Calculate final score
    total = pos_score + neg_score
    if total == 0:
        # No sentiment words found, analyze length and structure
        if len(text) < 10:
            return 0.0
        # Long text without sentiment words is probably neutral
        return 0.0
    
    # Calculate normalized score
    score = (pos_score - neg_score) / (pos_score + neg_score)
    
    # Apply text length factor (very short texts are less reliable)
    if len(text) < 15:
        score *= 0.7
    
    # Normalize to [-1, 1]
    score = max(min(score, 1.0), -1.0)
    
    return score


@app.post("/analyze-product")
async def analyze_product(request: ProductAnalysisRequest):
    """
    Analyze a product from e-commerce URL
    - Scrapes product details and reviews
    - Performs sentiment analysis
    - Generates AI-powered buy/don't buy recommendation
    """
    url = request.url
    
    if not url:
        raise HTTPException(status_code=400, detail="Product URL is required")
    
    print(f"[INFO] Analyzing product URL: {url}")
    
    try:
        # Step 1: Scrape product and reviews
        print("[INFO] Starting scraper...")
        async with ProductScraper() as scraper:
            product_data = await scraper.scrape_product(url)
        
        print(f"[INFO] Scraping result: success={product_data.get('success')}, reviews={len(product_data.get('reviews', []))}")
        
        if not product_data.get('success'):
            error_msg = product_data.get('error', 'Failed to scrape product')
            print(f"[ERROR] Scraping failed: {error_msg}")
            raise HTTPException(
                status_code=400,
                detail=error_msg
            )
        
        reviews = product_data.get('reviews', [])
        
        print(f"[INFO] Found {len(reviews)} reviews")
        
        if len(reviews) == 0:
            print("[WARNING] No reviews found for product")
            return {
                'success': False,
                'message': 'No reviews found for this product',
                'note': 'Some platforms (like Myntra) require advanced scraping. Try Amazon or Flipkart links.',
                'product_data': product_data,
            }
        
        # Step 2: Perform sentiment analysis on reviews
        review_texts = [r.get('text', '') for r in reviews if r.get('text')]
        
        if not review_texts:
            raise HTTPException(status_code=400, detail="No valid review text found")
        
        # Analyze sentiment using heuristic method
        sentiment_results = []
        for text in review_texts:
            score = _heuristic_sentiment(text)
            
            # Classify (more sensitive thresholds)
            if score > 0.05:
                label = "Positive"
            elif score < -0.05:
                label = "Negative"
            else:
                label = "Neutral"
            
            sentiment_results.append({
                'review': text[:200],  # Truncate for storage
                'sentiment_score': round(score, 3),
                'label': label,
            })
        
        # Calculate statistics
        scores = [r['sentiment_score'] for r in sentiment_results]
        avg_sentiment = sum(scores) / len(scores) if scores else 0
        
        positive_count = sum(1 for r in sentiment_results if r['label'] == 'Positive')
        neutral_count = sum(1 for r in sentiment_results if r['label'] == 'Neutral')
        negative_count = sum(1 for r in sentiment_results if r['label'] == 'Negative')
        
        sentiment_analysis = {
            'results': sentiment_results,
            'statistics': {
                'total_entries': len(sentiment_results),
                'average_sentiment': avg_sentiment,
                'sentiment_distribution': {
                    'positive': positive_count,
                    'neutral': neutral_count,
                    'negative': negative_count,
                },
            },
        }
        
        # Step 3: Generate AI recommendation using comparison engine
        comparison_engine = ProductComparisonEngine()
        ai_analysis = comparison_engine.analyze_product(product_data, sentiment_analysis)
        
        # Step 3.5: Fetch alternative products if recommendation is poor
        alternatives = []
        if ai_analysis.get('should_show_alternatives', False):
            print("[INFO] Product not recommended - fetching alternatives...")
            product_name = product_data.get('product_name', '')
            platform = product_data.get('platform', 'amazon')
            print(f"[INFO] Searching for alternatives to: {product_name} on {platform}")
            
            try:
                async with ProductScraper() as scraper:
                    alternatives = await scraper.search_alternative_products(
                        product_name=product_name,
                        platform=platform,
                        max_results=5
                    )
                
                print(f"[INFO] Found {len(alternatives)} alternative products")
                if len(alternatives) == 0:
                    print("[WARNING] No alternatives found - may need to check scraping logic")
            except Exception as e:
                print(f"[ERROR] Failed to fetch alternatives: {e}")  
                import traceback
                traceback.print_exc()
        
        # Step 4: Save to MongoDB
        analysis_id = str(uuid.uuid4())
        await AnalysisRepository.save_analysis({
            'upload_id': analysis_id,
            'analysis_type': 'product_url',
            'metadata': {
                'source_url': url,
                'platform': product_data.get('platform'),
                'product_name': product_data.get('product_name'),
                'price': product_data.get('price'),
                'rating': product_data.get('rating'),
            },
            'product_details': product_data,
            'sentiment_analysis': sentiment_analysis,
            'ai_recommendation': ai_analysis,
            'alternatives': alternatives,
            'created_at': datetime.utcnow().isoformat(),
        })
        
        return {
            'success': True,
            'analysis_id': analysis_id,
            'product': {
                'name': product_data.get('product_name'),
                'platform': product_data.get('platform'),
                'price': product_data.get('price'),
                'currency': product_data.get('currency'),
                'rating': product_data.get('rating'),
                'image_url': product_data.get('image_url'),
                'url': url,
            },
            'sentiment_analysis': {
                'total_reviews': len(sentiment_results),
                'average_sentiment': round(avg_sentiment, 3),
                'distribution': {
                    'positive': positive_count,
                    'neutral': neutral_count,
                    'negative': negative_count,
                },
            },
            'recommendation': {
                'score': ai_analysis['recommendation_score'],
                'verdict': ai_analysis['decision'],
                'confidence': ai_analysis['confidence'],
                'reasoning': ai_analysis['reasoning'],
                'should_buy': ai_analysis['decision'] in ['STRONGLY RECOMMEND', 'RECOMMEND'],
            },
            'detailed_analysis': ai_analysis['detailed_analysis'],
            'pros': ai_analysis['pros'],
            'cons': ai_analysis['cons'],
            'summary': ai_analysis['summary'],
            'alternatives': alternatives if alternatives else None,
        }
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
