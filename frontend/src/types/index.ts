export interface SentimentEntry {
    id: string;
    text: string;
    timestamp: string | null;
    sentiment_label: 'Positive' | 'Neutral' | 'Negative';
    sentiment_score: number;
    confidence: number;
    probabilities: {
        negative: number;
        neutral: number;
        positive: number;
    };
    tokens: string[];
    attention_weights: number[];
    raw_data?: Record<string, any>;
}

export interface DatasetMetadata {
    source_file: string;
    file_type: string;
    total_entries: number;
    detected_columns?: {
        text: string | null;
        timestamp: string | null;
        id: string | null;
    };
    all_columns?: string[];
    total_pages?: number;
    parsed_at: string;
}

export interface DatasetStatistics {
    total_entries: number;
    sentiment_distribution: {
        negative: number;
        neutral: number;
        positive: number;
    };
    average_sentiment: number;
    risk_score: number;
    confidence_avg: number;
}

export interface TrendAnalysis {
    average_sentiment: number;
    sentiment_std: number;
    trend_direction: 'improving' | 'declining' | 'stable' | 'unknown';
    momentum: string;
    distribution: {
        positive: number;
        neutral: number;
        negative: number;
        positive_pct: number;
        neutral_pct: number;
        negative_pct: number;
    };
    volatility: number;
    health_score: number;
    average_confidence: number;
    total_analyzed: number;
}

export interface ForecastData {
    forecasts: number[];
    upper_bound: number[];
    lower_bound: number[];
    time_labels: string[];
    trend_coefficient: number;
    confidence_level: number;
    future_distribution: Array<{
        positive: number;
        neutral: number;
        negative: number;
    }>;
    forecast_summary: string;
}

export interface Recommendation {
    priority: 'high' | 'medium' | 'low';
    category: string;
    title: string;
    description: string;
    action: string;
    impact: string;
}

export interface ProcessedDataset {
    upload_id: string;
    metadata: DatasetMetadata;
    data: SentimentEntry[];
    statistics: DatasetStatistics;
    trends?: TrendAnalysis;
    forecast?: ForecastData;
    recommendations?: Recommendation[];
    processed_at: string;
}

export interface PredictionResult {
    text: string;
    sentiment_label: 'Positive' | 'Neutral' | 'Negative';
    sentiment_score: number;
    confidence: number;
    probabilities: {
        negative: number;
        neutral: number;
        positive: number;
    };
    tokens: string[];
    attention_weights: number[];
}
