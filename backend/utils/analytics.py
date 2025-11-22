"""
Advanced Analytics Module for Sentiment Predictions and Recommendations
Provides trend analysis, forecasting, and AI-powered insights
"""

import numpy as np
from typing import List, Dict, Tuple, Any
from datetime import datetime, timedelta
from collections import Counter
import math


class SentimentAnalytics:
    """Advanced analytics for sentiment data with forecasting capabilities."""
    
    def __init__(self):
        self.sentiment_history = []
        self.time_series = []
    
    def analyze_trends(self, sentiment_data: List[Dict]) -> Dict[str, Any]:
        """
        Analyze sentiment trends and patterns.
        
        Args:
            sentiment_data: List of sentiment entries with scores
            
        Returns:
            Dictionary with trend analysis
        """
        if not sentiment_data:
            return self._empty_analysis()
        
        # Extract sentiment scores
        scores = [entry['sentiment_score'] for entry in sentiment_data]
        labels = [entry['sentiment_label'] for entry in sentiment_data]
        confidences = [entry['confidence'] for entry in sentiment_data]
        
        # Calculate basic statistics
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        trend_direction = self._calculate_trend(scores)
        
        # Calculate momentum (rate of change)
        momentum = self._calculate_momentum(scores)
        
        # Sentiment distribution
        label_counts = Counter(labels)
        total = len(labels)
        distribution = {
            'positive': label_counts.get('Positive', 0),
            'neutral': label_counts.get('Neutral', 0),
            'negative': label_counts.get('Negative', 0),
            'positive_pct': (label_counts.get('Positive', 0) / total) * 100,
            'neutral_pct': (label_counts.get('Neutral', 0) / total) * 100,
            'negative_pct': (label_counts.get('Negative', 0) / total) * 100
        }
        
        # Volatility (how stable the sentiment is)
        volatility = self._calculate_volatility(scores)
        
        # Overall health score (0-100)
        health_score = self._calculate_health_score(
            avg_score, distribution, volatility, np.mean(confidences)
        )
        
        return {
            'average_sentiment': float(avg_score),
            'sentiment_std': float(std_score),
            'trend_direction': trend_direction,
            'momentum': momentum,
            'distribution': distribution,
            'volatility': volatility,
            'health_score': health_score,
            'average_confidence': float(np.mean(confidences)),
            'total_analyzed': total
        }
    
    def forecast_sentiment(
        self, 
        sentiment_data: List[Dict], 
        periods: int = 12
    ) -> Dict[str, Any]:
        """
        Forecast sentiment for next 3-4 months using advanced time series analysis.
        Uses ARIMA-like approach with exponential smoothing and trend detection.
        
        Args:
            sentiment_data: Historical sentiment data
            periods: Number of future periods to forecast (weeks)
            
        Returns:
            Dictionary with forecasted values and insights
        """
        if len(sentiment_data) < 5:
            return self._default_forecast(periods)
        
        scores = np.array([entry['sentiment_score'] for entry in sentiment_data])
        n = len(scores)
        
        # Calculate multiple trend components
        linear_trend = self._calculate_linear_trend(scores)
        momentum = self._calculate_momentum_coefficient(scores)
        seasonality = self._detect_seasonality(scores)
        
        # Calculate moving averages for smoothing
        window = min(5, n // 2)
        if window >= 2:
            ma = np.convolve(scores, np.ones(window)/window, mode='valid')
            recent_avg = ma[-1] if len(ma) > 0 else scores[-1]
        else:
            recent_avg = np.mean(scores[-3:])
        
        # Exponential smoothing weights (more weight to recent data)
        alpha = 0.4  # Level smoothing
        beta = 0.3   # Trend smoothing
        
        # Initialize level and trend
        level = recent_avg
        trend = linear_trend
        
        forecasts = []
        confidence_decay = 0.95  # Confidence decreases over time
        
        for i in range(1, periods + 1):
            # Dampen trend over time (trends don't continue forever)
            dampen_factor = confidence_decay ** i
            
            # Apply exponential smoothing with trend
            forecast_value = level + (trend * i * dampen_factor)
            
            # Add momentum component (acceleration/deceleration)
            momentum_effect = momentum * (i ** 0.5) * dampen_factor * 0.5
            forecast_value += momentum_effect
            
            # Add seasonality if detected
            if seasonality['has_seasonality']:
                season_effect = seasonality['amplitude'] * np.sin(2 * np.pi * i / seasonality['period'])
                forecast_value += season_effect * dampen_factor
            
            # Mean reversion (sentiment tends to move towards average over time)
            mean_score = np.mean(scores)
            reversion_strength = min(0.1 * i, 0.4)  # Stronger over time
            forecast_value = forecast_value * (1 - reversion_strength) + mean_score * reversion_strength
            
            # Ensure realistic bounds
            forecast_value = np.clip(forecast_value, -1.0, 1.0)
            forecasts.append(float(forecast_value))
        
        # Calculate dynamic confidence intervals based on historical volatility
        historical_std = np.std(scores)
        volatility_factor = 1 + (historical_std / 0.5)  # Normalize volatility
        
        upper_bound = []
        lower_bound = []
        for i, forecast in enumerate(forecasts):
            # Widen intervals over time
            interval_width = historical_std * volatility_factor * (1 + i * 0.05)
            upper_bound.append(float(min(forecast + interval_width, 1.0)))
            lower_bound.append(float(max(forecast - interval_width, -1.0)))
        
        # Generate time labels
        time_labels = self._generate_time_labels(periods)
        
        # Predict future sentiment distribution
        future_dist = self._forecast_distribution_accurate(scores, forecasts)
        
        # Calculate forecast confidence based on data quality
        confidence = self._calculate_forecast_confidence_advanced(scores, linear_trend, historical_std)
        
        # Generate detailed forecast summary
        forecast_summary = self._generate_forecast_summary_detailed(
            forecasts, trend, momentum, confidence
        )
        
        return {
            'forecasts': forecasts,
            'upper_bound': upper_bound,
            'lower_bound': lower_bound,
            'time_labels': time_labels,
            'trend_coefficient': float(linear_trend),
            'confidence_level': confidence,
            'future_distribution': future_dist,
            'forecast_summary': forecast_summary,
            'momentum': float(momentum),
            'volatility': float(historical_std)
        }
    
    def generate_recommendations(
        self, 
        sentiment_data: List[Dict],
        forecast_data: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """
        Generate AI-powered recommendations based on analysis and forecast.
        
        Returns:
            List of recommendations with priority levels
        """
        recommendations = []
        
        # Analyze current state
        analysis = self.analyze_trends(sentiment_data)
        avg_sentiment = analysis['average_sentiment']
        trend = analysis['trend_direction']
        volatility = analysis['volatility']
        health_score = analysis['health_score']
        neg_pct = analysis['distribution']['negative_pct']
        
        # Future forecast analysis
        forecast_trend = forecast_data['trend_coefficient']
        future_sentiment = np.mean(forecast_data['forecasts'][:4])  # Next month
        
        # Critical issues (High Priority)
        if neg_pct > 40:
            recommendations.append({
                'priority': 'high',
                'category': 'Immediate Action Required',
                'title': 'High Negative Sentiment Detected',
                'description': f'{neg_pct:.1f}% of feedback is negative. Immediate intervention needed to address customer concerns.',
                'action': 'Review negative feedback, identify common issues, and implement quick fixes.',
                'impact': 'Critical'
            })
        
        if health_score < 40:
            recommendations.append({
                'priority': 'high',
                'category': 'Product Health',
                'title': 'Poor Overall Product Health',
                'description': f'Health score is {health_score:.0f}/100. Product perception is declining.',
                'action': 'Conduct thorough product review, gather detailed user feedback, and prioritize improvements.',
                'impact': 'Critical'
            })
        
        # Trend-based recommendations (Medium Priority)
        if trend == 'declining' and forecast_trend < -0.02:
            recommendations.append({
                'priority': 'medium',
                'category': 'Trend Alert',
                'title': 'Declining Sentiment Trend',
                'description': 'Sentiment is decreasing over time. Forecast shows continued decline if no action is taken.',
                'action': 'Launch improvement initiatives, enhance customer support, and communicate product updates.',
                'impact': 'High'
            })
        
        if volatility > 0.5:
            recommendations.append({
                'priority': 'medium',
                'category': 'Consistency',
                'title': 'High Sentiment Volatility',
                'description': 'User experience is inconsistent. Some love it, others don\'t.',
                'action': 'Standardize user experience, improve documentation, and provide better onboarding.',
                'impact': 'Medium'
            })
        
        # Positive trends (Low Priority - Optimization)
        if trend == 'improving' and avg_sentiment > 0.3:
            recommendations.append({
                'priority': 'low',
                'category': 'Growth Opportunity',
                'title': 'Positive Momentum Building',
                'description': 'Sentiment is improving. Capitalize on this positive trend.',
                'action': 'Increase marketing efforts, gather testimonials, and encourage user reviews.',
                'impact': 'Medium'
            })
        
        if health_score > 75 and neg_pct < 15:
            recommendations.append({
                'priority': 'low',
                'category': 'Success',
                'title': 'Strong Product Performance',
                'description': f'Health score: {health_score:.0f}/100. Users are highly satisfied.',
                'action': 'Maintain current quality, continue engaging with users, and plan feature expansion.',
                'impact': 'Low'
            })
        
        # Future forecast recommendations
        if future_sentiment < avg_sentiment - 0.15:
            recommendations.append({
                'priority': 'medium',
                'category': 'Forecast Warning',
                'title': 'Predicted Sentiment Decline',
                'description': 'AI models predict sentiment will decrease in coming weeks if current trends continue.',
                'action': 'Proactively address emerging issues, plan product updates, and improve communication.',
                'impact': 'High'
            })
        
        # Market positioning
        if 60 <= health_score <= 75:
            recommendations.append({
                'priority': 'low',
                'category': 'Optimization',
                'title': 'Room for Improvement',
                'description': 'Good performance but not exceptional. Opportunity to become market leader.',
                'action': 'Analyze competitor strategies, identify unique value propositions, and enhance key features.',
                'impact': 'Medium'
            })
        
        # Sort by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        recommendations.sort(key=lambda x: priority_order[x['priority']])
        
        return recommendations if recommendations else self._default_recommendations()
    
    def _calculate_trend(self, scores: List[float]) -> str:
        """Determine if trend is improving, declining, or stable."""
        if len(scores) < 3:
            return 'stable'
        
        # Compare first half vs second half
        mid = len(scores) // 2
        first_half = np.mean(scores[:mid])
        second_half = np.mean(scores[mid:])
        
        diff = second_half - first_half
        
        if diff > 0.1:
            return 'improving'
        elif diff < -0.1:
            return 'declining'
        else:
            return 'stable'
    
    def _calculate_trend_coefficient(self, scores: np.ndarray) -> float:
        """Calculate linear trend coefficient using least squares."""
        if len(scores) < 2:
            return 0.0
        
        x = np.arange(len(scores))
        z = np.polyfit(x, scores, 1)
        return float(z[0])
    
    def _calculate_linear_trend(self, scores: np.ndarray) -> float:
        """Calculate linear trend with robust estimation."""
        if len(scores) < 3:
            return 0.0
        
        # Use weighted linear regression (more weight to recent data)
        x = np.arange(len(scores))
        weights = np.exp(np.linspace(-1, 0, len(scores)))  # Exponential weights
        
        # Weighted mean
        wx = np.sum(weights * x) / np.sum(weights)
        wy = np.sum(weights * scores) / np.sum(weights)
        
        # Weighted covariance and variance
        cov = np.sum(weights * (x - wx) * (scores - wy)) / np.sum(weights)
        var = np.sum(weights * (x - wx) ** 2) / np.sum(weights)
        
        if var == 0:
            return 0.0
        
        slope = cov / var
        return float(slope)
    
    def _calculate_momentum_coefficient(self, scores: np.ndarray) -> float:
        """Calculate momentum (rate of change acceleration)."""
        if len(scores) < 5:
            return 0.0
        
        # Calculate first derivative (velocity)
        velocity = np.diff(scores)
        
        # Calculate second derivative (acceleration)
        acceleration = np.diff(velocity)
        
        # Return recent acceleration
        recent_accel = np.mean(acceleration[-3:]) if len(acceleration) >= 3 else np.mean(acceleration)
        return float(recent_accel)
    
    def _detect_seasonality(self, scores: np.ndarray) -> Dict[str, Any]:
        """Detect if there's any seasonal pattern in the data."""
        if len(scores) < 8:
            return {'has_seasonality': False, 'period': 0, 'amplitude': 0}
        
        # Simple autocorrelation check for common periods
        max_period = min(len(scores) // 2, 7)
        best_correlation = 0
        best_period = 0
        
        for period in range(2, max_period + 1):
            correlation = 0
            count = 0
            for i in range(len(scores) - period):
                correlation += scores[i] * scores[i + period]
                count += 1
            
            if count > 0:
                correlation /= count
                if abs(correlation) > abs(best_correlation):
                    best_correlation = correlation
                    best_period = period
        
        # If correlation is significant, we have seasonality
        has_seasonality = abs(best_correlation) > 0.3
        amplitude = np.std(scores) * abs(best_correlation) if has_seasonality else 0
        
        return {
            'has_seasonality': has_seasonality,
            'period': best_period,
            'amplitude': float(amplitude)
        }
    
    def _forecast_distribution_accurate(self, historical: np.ndarray, forecasts: List[float]) -> List[Dict]:
        """Forecast distribution based on historical patterns and predicted scores."""
        future_dist = []
        
        # Analyze historical distribution at different sentiment levels
        for forecast in forecasts:
            # Find similar historical scores
            tolerance = 0.2
            similar_mask = np.abs(historical - forecast) < tolerance
            
            if np.sum(similar_mask) > 0:
                # Use historical distribution at similar sentiment levels
                # This is a simplified version - in reality, we'd use the actual labels
                if forecast > 0.3:
                    pos, neu, neg = 0.65, 0.25, 0.10
                elif forecast > 0.1:
                    pos, neu, neg = 0.50, 0.35, 0.15
                elif forecast > -0.1:
                    pos, neu, neg = 0.30, 0.45, 0.25
                elif forecast > -0.3:
                    pos, neu, neg = 0.15, 0.35, 0.50
                else:
                    pos, neu, neg = 0.10, 0.25, 0.65
            else:
                # Default distribution based on forecast value
                if forecast > 0:
                    ratio = forecast
                    pos = 0.33 + ratio * 0.4
                    neg = 0.33 - ratio * 0.2
                    neu = 1 - pos - neg
                else:
                    ratio = abs(forecast)
                    neg = 0.33 + ratio * 0.4
                    pos = 0.33 - ratio * 0.2
                    neu = 1 - pos - neg
            
            future_dist.append({
                'positive': float(max(0, min(1, pos))),
                'neutral': float(max(0, min(1, neu))),
                'negative': float(max(0, min(1, neg)))
            })
        
        return future_dist
    
    def _calculate_forecast_confidence_advanced(
        self, 
        scores: np.ndarray,
        trend: float,
        volatility: float
    ) -> float:
        """Calculate confidence in forecast based on multiple factors."""
        # Data quantity confidence (more data = more confident)
        data_confidence = min(len(scores) / 30, 1.0) * 30
        
        # Volatility confidence (lower volatility = more confident)
        volatility_normalized = min(volatility / 0.5, 1.0)
        stability_confidence = (1 - volatility_normalized) * 35
        
        # Trend confidence (clear trend = more confident)
        trend_strength = min(abs(trend) / 0.05, 1.0)
        trend_confidence = trend_strength * 20
        
        # Consistency confidence (similar recent values = more confident)
        if len(scores) >= 5:
            recent_std = np.std(scores[-5:])
            consistency = 1 - min(recent_std / 0.5, 1.0)
            consistency_confidence = consistency * 15
        else:
            consistency_confidence = 5
        
        total_confidence = data_confidence + stability_confidence + trend_confidence + consistency_confidence
        return float(min(total_confidence, 95))  # Cap at 95%
    
    def _generate_forecast_summary_detailed(
        self,
        forecasts: List[float],
        trend: float,
        momentum: float,
        confidence: float
    ) -> str:
        """Generate detailed human-readable forecast summary."""
        avg_forecast = np.mean(forecasts[:4])  # Next month
        
        # Trend analysis
        if trend > 0.03:
            trend_desc = "strong upward trend"
        elif trend > 0.01:
            trend_desc = "moderate upward trend"
        elif trend < -0.03:
            trend_desc = "strong downward trend"
        elif trend < -0.01:
            trend_desc = "moderate downward trend"
        else:
            trend_desc = "stable trend"
        
        # Momentum analysis
        if momentum > 0.01:
            momentum_desc = "accelerating positively"
        elif momentum < -0.01:
            momentum_desc = "decelerating"
        else:
            momentum_desc = "steady"
        
        # Overall outlook
        if avg_forecast > 0.4:
            outlook = "strong positive"
        elif avg_forecast > 0.2:
            outlook = "positive"
        elif avg_forecast > -0.2:
            outlook = "neutral"
        elif avg_forecast > -0.4:
            outlook = "negative"
        else:
            outlook = "concerning"
        
        # Confidence qualifier
        if confidence > 75:
            conf_desc = "high confidence"
        elif confidence > 50:
            conf_desc = "moderate confidence"
        else:
            conf_desc = "low confidence (limited data)"
        
        return f"AI forecast shows a {trend_desc} with {momentum_desc} momentum. Expected {outlook} sentiment over the next 3 months ({conf_desc})."
    
    def _calculate_momentum(self, scores: List[float]) -> str:
        """Calculate sentiment momentum."""
        if len(scores) < 5:
            return 'neutral'
        
        recent = np.mean(scores[-3:])
        previous = np.mean(scores[-6:-3] if len(scores) >= 6 else scores[:-3])
        
        diff = recent - previous
        
        if diff > 0.15:
            return 'strong_positive'
        elif diff > 0.05:
            return 'positive'
        elif diff < -0.15:
            return 'strong_negative'
        elif diff < -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    def _calculate_volatility(self, scores: List[float]) -> float:
        """Calculate sentiment volatility (0-1)."""
        if len(scores) < 2:
            return 0.0
        
        # Calculate standard deviation and normalize
        std = np.std(scores)
        # Normalize to 0-1 range (std of 0.5 is high volatility)
        volatility = min(std / 0.5, 1.0)
        return float(volatility)
    
    def _calculate_health_score(
        self, 
        avg_sentiment: float, 
        distribution: Dict,
        volatility: float,
        avg_confidence: float
    ) -> float:
        """Calculate overall health score (0-100)."""
        # Sentiment component (40%)
        sentiment_score = ((avg_sentiment + 1) / 2) * 40
        
        # Distribution component (30%)
        pos_pct = distribution['positive_pct']
        neg_pct = distribution['negative_pct']
        dist_score = (pos_pct - neg_pct + 100) / 200 * 30
        
        # Stability component (20%)
        stability_score = (1 - volatility) * 20
        
        # Confidence component (10%)
        confidence_score = avg_confidence * 10
        
        total = sentiment_score + dist_score + stability_score + confidence_score
        return float(np.clip(total, 0, 100))
    
    def _get_distribution(self, sentiment_data: List[Dict]) -> Dict:
        """Get current sentiment distribution."""
        labels = [entry['sentiment_label'] for entry in sentiment_data]
        total = len(labels)
        return {
            'positive': labels.count('Positive') / total,
            'neutral': labels.count('Neutral') / total,
            'negative': labels.count('Negative') / total
        }
    
    def _forecast_distribution(self, current_dist: Dict, forecasts: List[float]) -> List[Dict]:
        """Forecast sentiment distribution for future periods."""
        future_dist = []
        
        for forecast in forecasts:
            # Map continuous score to distribution
            if forecast > 0.3:
                pos, neu, neg = 0.6, 0.3, 0.1
            elif forecast > 0:
                pos, neu, neg = 0.4, 0.4, 0.2
            elif forecast > -0.3:
                pos, neu, neg = 0.2, 0.5, 0.3
            else:
                pos, neu, neg = 0.1, 0.3, 0.6
            
            future_dist.append({
                'positive': pos,
                'neutral': neu,
                'negative': neg
            })
        
        return future_dist
    
    def _generate_time_labels(self, periods: int) -> List[str]:
        """Generate time labels for forecast periods."""
        labels = []
        current = datetime.now()
        
        for i in range(1, periods + 1):
            future_date = current + timedelta(weeks=i)
            labels.append(future_date.strftime('%b %d'))
        
        return labels
    
    def _calculate_forecast_confidence(self, scores: np.ndarray) -> float:
        """Calculate confidence in forecast (0-100)."""
        # More data = higher confidence
        data_confidence = min(len(scores) / 50, 1.0) * 40
        
        # Lower volatility = higher confidence
        volatility = self._calculate_volatility(scores.tolist())
        stability_confidence = (1 - volatility) * 60
        
        return float(data_confidence + stability_confidence)
    
    def _generate_forecast_summary(self, forecasts: List[float], trend: float) -> str:
        """Generate human-readable forecast summary."""
        avg_future = np.mean(forecasts[:4])
        
        if trend > 0.02:
            direction = "improving trend"
        elif trend < -0.02:
            direction = "declining trend"
        else:
            direction = "stable trend"
        
        if avg_future > 0.3:
            outlook = "positive"
        elif avg_future > 0:
            outlook = "neutral to positive"
        elif avg_future > -0.3:
            outlook = "neutral to negative"
        else:
            outlook = "negative"
        
        return f"Forecast shows a {direction} with {outlook} sentiment expected in the coming weeks."
    
    def _empty_analysis(self) -> Dict:
        """Return empty analysis structure."""
        return {
            'average_sentiment': 0.0,
            'sentiment_std': 0.0,
            'trend_direction': 'unknown',
            'momentum': 'neutral',
            'distribution': {'positive': 0, 'neutral': 0, 'negative': 0},
            'volatility': 0.0,
            'health_score': 50.0,
            'average_confidence': 0.0,
            'total_analyzed': 0
        }
    
    def _default_forecast(self, periods: int) -> Dict:
        """Return default forecast when insufficient data."""
        return {
            'forecasts': [0.0] * periods,
            'upper_bound': [0.3] * periods,
            'lower_bound': [-0.3] * periods,
            'time_labels': self._generate_time_labels(periods),
            'trend_coefficient': 0.0,
            'confidence_level': 30.0,
            'future_distribution': [{'positive': 0.33, 'neutral': 0.34, 'negative': 0.33}] * periods,
            'forecast_summary': 'Insufficient data for accurate forecasting. Upload more data for better predictions.'
        }
    
    def _default_recommendations(self) -> List[Dict]:
        """Return default recommendations."""
        return [{
            'priority': 'medium',
            'category': 'Data Collection',
            'title': 'Gather More Data',
            'description': 'Insufficient data to generate specific recommendations.',
            'action': 'Continue collecting user feedback and reviews to enable AI-powered insights.',
            'impact': 'Medium'
        }]
