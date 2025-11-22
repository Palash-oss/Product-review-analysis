"""
AI-Powered Product Comparison and Recommendation Engine
Analyzes sentiment, compares products, and generates purchase recommendations
"""

from typing import Dict, List, Tuple
import re
from collections import Counter
import statistics


class ProductComparisonEngine:
    """Intelligent product comparison and recommendation system"""
    
    def __init__(self):
        # Decision factors and weights
        self.weights = {
            'sentiment_score': 0.35,      # 35% weight on overall sentiment
            'rating': 0.25,               # 25% weight on numeric rating
            'review_quality': 0.15,       # 15% weight on review detail/quality
            'price_value': 0.15,          # 15% weight on price-to-value ratio
            'negative_flags': 0.10,       # 10% weight on critical issues
        }
        
        # Critical negative keywords that impact buying decision
        self.critical_keywords = {
            'defect': 3.0,
            'broken': 3.0,
            'fake': 3.5,
            'scam': 3.5,
            'fraud': 3.5,
            'damaged': 2.5,
            'poor quality': 2.5,
            'waste': 2.0,
            'terrible': 2.0,
            'worst': 2.0,
            'useless': 2.0,
            'disappointed': 1.5,
            'not worth': 1.8,
            'regret': 1.8,
            'refund': 1.5,
            'return': 1.3,
        }
        
        # Positive indicators
        self.positive_keywords = {
            'excellent': 2.5,
            'amazing': 2.5,
            'perfect': 2.5,
            'highly recommend': 3.0,
            'worth': 2.0,
            'best': 2.0,
            'love': 1.8,
            'great': 1.5,
            'good': 1.2,
            'satisfied': 1.5,
        }
    
    def analyze_product(self, product_data: Dict, sentiment_results: Dict) -> Dict:
        """
        Comprehensive product analysis with buy/don't buy recommendation
        
        Args:
            product_data: Scraped product data from scraper
            sentiment_results: Sentiment analysis results from ML model
        
        Returns:
            Complete analysis with recommendation and alternatives
        """
        reviews = product_data.get('reviews', [])
        
        # Calculate comprehensive scores
        sentiment_analysis = self._analyze_sentiment_distribution(sentiment_results)
        review_analysis = self._analyze_review_quality(reviews)
        risk_analysis = self._analyze_risks(reviews, sentiment_results)
        price_analysis = self._analyze_price_value(product_data, sentiment_analysis)
        
        # Calculate final recommendation score (0-100)
        recommendation_score = self._calculate_recommendation_score({
            'sentiment': sentiment_analysis,
            'reviews': review_analysis,
            'risks': risk_analysis,
            'price': price_analysis,
        })
        
        # Generate decision
        decision = self._generate_decision(recommendation_score, risk_analysis)
        
        # Extract key insights
        pros = self._extract_pros(reviews, sentiment_results)
        cons = self._extract_cons(reviews, sentiment_results)
        
        # Determine if alternatives should be recommended
        should_show_alternatives = recommendation_score < 55 or risk_analysis['level'] in ['High', 'Critical']
        
        return {
            'recommendation_score': round(recommendation_score, 2),
            'decision': decision['verdict'],
            'confidence': decision['confidence'],
            'reasoning': decision['reasoning'],
            'should_show_alternatives': should_show_alternatives,
            'detailed_analysis': {
                'sentiment_breakdown': sentiment_analysis,
                'review_quality': review_analysis,
                'risk_factors': risk_analysis,
                'price_analysis': price_analysis,
            },
            'pros': pros[:10],  # Top 10 pros
            'cons': cons[:10],  # Top 10 cons
            'summary': self._generate_summary(
                product_data,
                recommendation_score,
                decision,
                pros,
                cons
            ),
        }
    
    def _analyze_sentiment_distribution(self, sentiment_results: Dict) -> Dict:
        """Analyze sentiment distribution from ML model"""
        distribution = sentiment_results.get('statistics', {}).get('sentiment_distribution', {})
        total = distribution.get('positive', 0) + distribution.get('neutral', 0) + distribution.get('negative', 0)
        
        if total == 0:
            return {'score': 0, 'grade': 'F', 'description': 'No reviews available'}
        
        # Calculate weighted sentiment score
        positive_pct = (distribution.get('positive', 0) / total) * 100
        neutral_pct = (distribution.get('neutral', 0) / total) * 100
        negative_pct = (distribution.get('negative', 0) / total) * 100
        
        # Weighted score (positive=1, neutral=0.5, negative=0)
        sentiment_score = (positive_pct + (neutral_pct * 0.5)) / 100
        
        # Average sentiment value
        avg_sentiment = sentiment_results.get('statistics', {}).get('average_sentiment', 0)
        
        # Determine grade
        if sentiment_score >= 0.8:
            grade = 'A+'
        elif sentiment_score >= 0.7:
            grade = 'A'
        elif sentiment_score >= 0.6:
            grade = 'B+'
        elif sentiment_score >= 0.5:
            grade = 'B'
        elif sentiment_score >= 0.4:
            grade = 'C'
        elif sentiment_score >= 0.3:
            grade = 'D'
        else:
            grade = 'F'
        
        return {
            'score': sentiment_score,
            'grade': grade,
            'positive_pct': round(positive_pct, 1),
            'neutral_pct': round(neutral_pct, 1),
            'negative_pct': round(negative_pct, 1),
            'average_sentiment': avg_sentiment,
            'description': self._describe_sentiment(sentiment_score),
        }
    
    def _analyze_review_quality(self, reviews: List[Dict]) -> Dict:
        """Analyze quality and depth of reviews"""
        if not reviews:
            return {'score': 0, 'quality': 'Poor', 'details': 'No reviews available'}
        
        # Calculate metrics
        total_reviews = len(reviews)
        avg_length = statistics.mean([len(r.get('text', '')) for r in reviews])
        verified_count = sum(1 for r in reviews if r.get('verified_purchase', False))
        verified_pct = (verified_count / total_reviews) * 100 if total_reviews > 0 else 0
        
        # Reviews with ratings
        rated_reviews = [r for r in reviews if r.get('rating') is not None]
        avg_rating = statistics.mean([r['rating'] for r in rated_reviews]) if rated_reviews else 0
        
        # Quality score calculation
        quality_score = 0
        
        # Factor 1: Number of reviews (0-0.3)
        if total_reviews >= 100:
            quality_score += 0.3
        elif total_reviews >= 50:
            quality_score += 0.25
        elif total_reviews >= 20:
            quality_score += 0.2
        elif total_reviews >= 10:
            quality_score += 0.15
        else:
            quality_score += 0.1
        
        # Factor 2: Average review length (0-0.3)
        if avg_length >= 200:
            quality_score += 0.3
        elif avg_length >= 100:
            quality_score += 0.25
        elif avg_length >= 50:
            quality_score += 0.15
        else:
            quality_score += 0.1
        
        # Factor 3: Verified purchases (0-0.4)
        quality_score += (verified_pct / 100) * 0.4
        
        return {
            'score': quality_score,
            'total_reviews': total_reviews,
            'avg_length': round(avg_length, 1),
            'verified_pct': round(verified_pct, 1),
            'avg_rating': round(avg_rating, 2),
            'quality': self._rate_quality(quality_score),
        }
    
    def _analyze_risks(self, reviews: List[Dict], sentiment_results: Dict) -> Dict:
        """Identify and quantify risk factors"""
        risk_score = 0
        risk_factors = []
        
        # Check for critical keywords in reviews
        all_review_text = ' '.join([r.get('text', '').lower() for r in reviews])
        
        critical_mentions = {}
        for keyword, weight in self.critical_keywords.items():
            count = all_review_text.count(keyword)
            if count > 0:
                critical_mentions[keyword] = count
                risk_score += count * weight
        
        # Risk factor: High negative percentage
        neg_pct = sentiment_results.get('statistics', {}).get('sentiment_distribution', {}).get('negative', 0)
        total = sum(sentiment_results.get('statistics', {}).get('sentiment_distribution', {}).values())
        if total > 0:
            neg_percentage = (neg_pct / total) * 100
            if neg_percentage > 50:
                risk_factors.append(f"High negative sentiment: {neg_percentage:.1f}%")
                risk_score += 20
            elif neg_percentage > 30:
                risk_factors.append(f"Moderate negative sentiment: {neg_percentage:.1f}%")
                risk_score += 10
        
        # Risk factor: Low rating
        avg_rating = sentiment_results.get('statistics', {}).get('average_sentiment', 0)
        if avg_rating < -0.3:
            risk_factors.append("Very low average sentiment score")
            risk_score += 15
        elif avg_rating < 0:
            risk_factors.append("Below-average sentiment score")
            risk_score += 8
        
        # Risk factor: Few reviews
        if len(reviews) < 5:
            risk_factors.append("Very few reviews (limited data)")
            risk_score += 10
        elif len(reviews) < 15:
            risk_factors.append("Limited number of reviews")
            risk_score += 5
        
        # Add specific critical mentions
        for keyword, count in sorted(critical_mentions.items(), key=lambda x: x[1], reverse=True)[:5]:
            risk_factors.append(f"'{keyword}' mentioned {count} time(s)")
        
        return {
            'risk_score': min(risk_score, 100),  # Cap at 100
            'level': self._rate_risk(risk_score),
            'factors': risk_factors[:10],  # Top 10 factors
            'critical_keywords_found': len(critical_mentions),
        }
    
    def _analyze_price_value(self, product_data: Dict, sentiment_analysis: Dict) -> Dict:
        """Analyze price-to-value ratio"""
        price = product_data.get('price')
        sentiment_score = sentiment_analysis.get('score', 0)
        
        if not price:
            return {
                'value_score': sentiment_score,
                'rating': 'Unknown',
                'description': 'Price information not available',
            }
        
        # Value score = sentiment quality relative to price expectations
        # Assume: Higher price = higher expectations
        
        if price < 500:
            price_tier = 'budget'
            expected_sentiment = 0.6  # Lower expectations
        elif price < 2000:
            price_tier = 'mid-range'
            expected_sentiment = 0.7
        elif price < 10000:
            price_tier = 'premium'
            expected_sentiment = 0.75
        else:
            price_tier = 'luxury'
            expected_sentiment = 0.85  # High expectations
        
        # Value score: How well it meets expectations
        value_score = sentiment_score / expected_sentiment if expected_sentiment > 0 else sentiment_score
        value_score = min(value_score, 1.0)  # Cap at 1.0
        
        if value_score >= 0.9:
            rating = 'Excellent Value'
        elif value_score >= 0.75:
            rating = 'Good Value'
        elif value_score >= 0.6:
            rating = 'Fair Value'
        elif value_score >= 0.4:
            rating = 'Below Average Value'
        else:
            rating = 'Poor Value'
        
        return {
            'value_score': value_score,
            'price': price,
            'price_tier': price_tier,
            'rating': rating,
            'sentiment_vs_expected': f"{(value_score * 100):.1f}% of expectations met",
        }
    
    def _calculate_recommendation_score(self, analysis: Dict) -> float:
        """Calculate final recommendation score (0-100)"""
        sentiment_data = analysis['sentiment']
        review_data = analysis['reviews']
        risk_data = analysis['risks']
        price_data = analysis['price']
        
        # Component scores (0-100 scale)
        sentiment_component = sentiment_data['score'] * 100
        rating_component = sentiment_data.get('average_sentiment', 0) * 50 + 50  # Map -1:1 to 0:100
        review_quality_component = review_data['score'] * 100
        price_value_component = price_data['value_score'] * 100
        risk_penalty = risk_data['risk_score']  # Already 0-100, subtract it
        
        # Weighted average
        score = (
            sentiment_component * self.weights['sentiment_score'] +
            rating_component * self.weights['rating'] +
            review_quality_component * self.weights['review_quality'] +
            price_value_component * self.weights['price_value']
        )
        
        # Apply risk penalty
        score = score * (1 - (risk_penalty / 100) * self.weights['negative_flags'])
        
        return max(0, min(100, score))  # Clamp to 0-100
    
    def _generate_decision(self, score: float, risk_analysis: Dict) -> Dict:
        """Generate buy/don't buy decision with reasoning"""
        risk_level = risk_analysis['level']
        risk_score = risk_analysis['risk_score']
        
        # Decision logic
        if score >= 75 and risk_level in ['Low', 'Minimal']:
            verdict = 'STRONGLY RECOMMEND'
            confidence = 'Very High'
            reasoning = "Excellent product with high customer satisfaction and minimal risks."
        
        elif score >= 65 and risk_level in ['Low', 'Minimal', 'Moderate']:
            verdict = 'RECOMMEND'
            confidence = 'High'
            reasoning = "Good product with positive reviews. Minor concerns exist but overall solid choice."
        
        elif score >= 55 and risk_level != 'Critical':
            verdict = 'CAUTIOUSLY RECOMMEND'
            confidence = 'Moderate'
            reasoning = "Decent product but has some issues. Consider alternatives before purchasing."
        
        elif score >= 45:
            verdict = 'NEUTRAL'
            confidence = 'Low'
            reasoning = "Mixed reviews. Research thoroughly and check for better alternatives."
        
        elif score >= 35:
            verdict = 'NOT RECOMMENDED'
            confidence = 'Moderate'
            reasoning = "Below average product with significant negative feedback. Look for alternatives."
        
        else:
            verdict = 'STRONGLY NOT RECOMMENDED'
            confidence = 'High'
            reasoning = "Poor product quality with many complaints. Avoid purchasing."
        
        # Adjust for critical risks
        if risk_level == 'Critical' and verdict in ['RECOMMEND', 'STRONGLY RECOMMEND']:
            verdict = 'CAUTIOUSLY RECOMMEND'
            confidence = 'Low'
            reasoning += " However, critical issues have been reported by customers."
        
        return {
            'verdict': verdict,
            'confidence': confidence,
            'reasoning': reasoning,
            'should_buy': score >= 55 and risk_level != 'Critical',
        }
    
    def _extract_pros(self, reviews: List[Dict], sentiment_results: Dict) -> List[str]:
        """Extract top pros/positive aspects"""
        pros = []
        
        # Get positive reviews
        positive_reviews = [
            r for r in sentiment_results.get('results', [])
            if r.get('label') == 'Positive'
        ]
        
        # Extract common positive phrases
        positive_phrases = []
        for review in positive_reviews[:30]:  # Top 30 positive reviews
            text = review.get('review', '').lower()
            
            # Check for positive keywords
            for keyword in self.positive_keywords:
                if keyword in text:
                    # Extract sentence containing keyword
                    sentences = text.split('.')
                    for sentence in sentences:
                        if keyword in sentence and len(sentence) > 20:
                            positive_phrases.append(sentence.strip().capitalize())
                            break
        
        # Remove duplicates and return top pros
        seen = set()
        for phrase in positive_phrases:
            if phrase not in seen and len(phrase) > 15:
                pros.append(phrase)
                seen.add(phrase)
                if len(pros) >= 10:
                    break
        
        return pros
    
    def _extract_cons(self, reviews: List[Dict], sentiment_results: Dict) -> List[str]:
        """Extract top cons/negative aspects"""
        cons = []
        
        # Get negative reviews
        negative_reviews = [
            r for r in sentiment_results.get('results', [])
            if r.get('label') == 'Negative'
        ]
        
        # Extract common negative phrases
        negative_phrases = []
        for review in negative_reviews[:30]:  # Top 30 negative reviews
            text = review.get('review', '').lower()
            
            # Check for critical keywords
            for keyword in self.critical_keywords:
                if keyword in text:
                    # Extract sentence containing keyword
                    sentences = text.split('.')
                    for sentence in sentences:
                        if keyword in sentence and len(sentence) > 20:
                            negative_phrases.append(sentence.strip().capitalize())
                            break
        
        # Remove duplicates and return top cons
        seen = set()
        for phrase in negative_phrases:
            if phrase not in seen and len(phrase) > 15:
                cons.append(phrase)
                seen.add(phrase)
                if len(cons) >= 10:
                    break
        
        return cons
    
    def _generate_summary(
        self,
        product_data: Dict,
        score: float,
        decision: Dict,
        pros: List[str],
        cons: List[str]
    ) -> str:
        """Generate human-readable summary"""
        product_name = product_data.get('product_name', 'this product')
        platform = product_data.get('platform', 'the platform')
        
        summary = f"**{product_name}** on {platform} has received a recommendation score of **{score:.1f}/100**. "
        summary += f"\n\n**Decision: {decision['verdict']}** (Confidence: {decision['confidence']})\n\n"
        summary += f"{decision['reasoning']}\n\n"
        
        if pros:
            summary += "**Key Strengths:**\n"
            for pro in pros[:5]:
                summary += f"✓ {pro}\n"
            summary += "\n"
        
        if cons:
            summary += "**Key Concerns:**\n"
            for con in cons[:5]:
                summary += f"✗ {con}\n"
        
        return summary
    
    # Helper methods
    def _describe_sentiment(self, score: float) -> str:
        if score >= 0.8:
            return "Overwhelmingly Positive"
        elif score >= 0.7:
            return "Very Positive"
        elif score >= 0.6:
            return "Mostly Positive"
        elif score >= 0.5:
            return "Mixed"
        elif score >= 0.4:
            return "Mostly Negative"
        elif score >= 0.3:
            return "Very Negative"
        else:
            return "Overwhelmingly Negative"
    
    def _rate_quality(self, score: float) -> str:
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.65:
            return "Good"
        elif score >= 0.5:
            return "Average"
        elif score >= 0.35:
            return "Below Average"
        else:
            return "Poor"
    
    def _rate_risk(self, score: float) -> str:
        if score >= 50:
            return "Critical"
        elif score >= 30:
            return "High"
        elif score >= 15:
            return "Moderate"
        elif score >= 5:
            return "Low"
        else:
            return "Minimal"
