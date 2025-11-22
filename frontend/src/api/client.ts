import axios from 'axios';
import type { ProcessedDataset, PredictionResult } from '../types';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

export const uploadFile = async (file: File): Promise<ProcessedDataset> => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await axios.post(`${API_BASE_URL}/upload`, formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    });

    return response.data;
};

export const getDataset = async (uploadId: string): Promise<ProcessedDataset> => {
    const response = await api.get(`/dataset/${uploadId}`);
    return response.data;
};

export const predictSingle = async (text: string): Promise<PredictionResult> => {
    const response = await api.post('/predict', { text });
    return response.data;
};

export const predictBatch = async (texts: string[]): Promise<{ predictions: PredictionResult[] }> => {
    const response = await api.post('/predict/batch', { texts });
    return response.data;
};

export const checkHealth = async (): Promise<any> => {
    const response = await api.get('/');
    return response.data;
};
