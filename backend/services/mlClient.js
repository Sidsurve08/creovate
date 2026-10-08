/**
 * ML Service Client
 * Handles communication with ML service v2 endpoints with fallbacks
 */

const axios = require('axios');

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';
const TIMEOUT = 5000; // 5 second timeout
const MAX_RETRIES = 1;

class MLClient {
  constructor() {
    this.client = axios.create({
      baseURL: ML_SERVICE_URL,
      timeout: TIMEOUT,
      headers: {
        'Content-Type': 'application/json'
      }
    });
  }

  /**
   * Make request with retry logic
   */
  async request(config, retries = MAX_RETRIES) {
    try {
      const response = await this.client.request(config);
      return response.data;
    } catch (error) {
      if (retries > 0 && this.isRetryableError(error)) {
        console.log(`Retrying ML request (${MAX_RETRIES - retries + 1}/${MAX_RETRIES})...`);
        return this.request(config, retries - 1);
      }
      throw error;
    }
  }

  /**
   * Check if error is retryable
   */
  isRetryableError(error) {
    if (!error.response) {
      // Network errors are retryable
      return true;
    }
    const status = error.response.status;
    return status >= 500 || status === 429;
  }

  /**
   * Classify text into category
   */
  async classifyText(text) {
    try {
      const response = await this.request({
        method: 'POST',
        url: '/v2/classify/text',
        data: { text }
      });
      return response;
    } catch (error) {
      console.error('ML service unavailable for text classification, using fallback');
      // Return fallback response
      return this.fallbackClassifyText(text);
    }
  }

  /**
   * Fallback text classification (rule-based)
   */
  fallbackClassifyText(text) {
    const textLower = text.toLowerCase();
    
    const categoryKeywords = {
      pottery: ['pottery', 'clay', 'ceramic', 'wheel', 'kiln'],
      textiles: ['textile', 'fabric', 'weaving', 'cloth', 'silk', 'cotton'],
      paintings: ['painting', 'art', 'canvas', 'watercolor', 'oil'],
      bamboo_cane: ['bamboo', 'cane', 'basket', 'wicker'],
      wood: ['wood', 'carving', 'sculpture', 'timber'],
      metalwork: ['metal', 'brass', 'copper', 'iron', 'forging'],
      jewellery: ['jewelry', 'jewellery', 'gold', 'silver', 'gem'],
      leather: ['leather', 'bag', 'shoe', 'wallet'],
      stone_terracotta: ['stone', 'terracotta', 'sculpture'],
      toys: ['toy', 'doll', 'game'],
      embroidery: ['embroidery', 'stitch', 'needlework'],
      paper_scroll: ['paper', 'scroll', 'calligraphy'],
      other: []
    };

    let bestCategory = 'other';
    let maxMatches = 0;

    for (const [category, keywords] of Object.entries(categoryKeywords)) {
      const matches = keywords.filter(keyword => textLower.includes(keyword)).length;
      if (matches > maxMatches) {
        maxMatches = matches;
        bestCategory = category;
      }
    }

    const confidence = Math.min(0.5 + (maxMatches * 0.1), 0.9);

    return {
      category: bestCategory,
      confidence,
      source: 'rule_based_fallback'
    };
  }

  /**
   * Suggest price range
   */
  async suggestPrice(params) {
    try {
      const response = await this.request({
        method: 'POST',
        url: '/v2/price/suggest',
        data: params
      });
      return response;
    } catch (error) {
      console.error('ML service unavailable for price suggestion, using fallback');
      return this.fallbackSuggestPrice(params);
    }
  }

  /**
   * Fallback price suggestion (rule-based)
   */
  fallbackSuggestPrice(params) {
    const basePrices = {
      pottery: 500,
      textiles: 2000,
      paintings: 3000,
      bamboo_cane: 800,
      wood: 1500,
      metalwork: 2500,
      jewellery: 5000,
      leather: 1200,
      stone_terracotta: 1000,
      toys: 300,
      embroidery: 1800,
      paper_scroll: 600,
      other: 1000
    };

    const category = params.category || 'other';
    const basePrice = basePrices[category] || 1000;
    const complexity = params.complexity || 1;

    const suggested = basePrice * complexity;
    const minPrice = suggested * 0.7;
    const maxPrice = suggested * 1.3;

    return {
      min_price: minPrice,
      suggested_price: suggested,
      max_price: maxPrice,
      confidence: 0.5,
      source: 'rule_based_fallback'
    };
  }

  /**
   * Semantic search
   */
  async semanticSearch(query, options = {}) {
    try {
      const response = await this.request({
        method: 'POST',
        url: '/v2/search/semantic',
        data: {
          query,
          top_k: options.top_k || 10,
          index: options.index || 'faiss_listings'
        }
      });
      return response;
    } catch (error) {
      console.error('ML service unavailable for semantic search, using fallback');
      return {
        results: [],
        source: 'rule_based_fallback'
      };
    }
  }

  /**
   * Get model status
   */
  async getModelStatus() {
    try {
      const response = await this.request({
        method: 'GET',
        url: '/v2/models/status'
      });
      return response;
    } catch (error) {
      console.error('ML service unavailable for status check');
      return {
        models: {
          text_classifier: 'unavailable',
          price_model: 'unavailable',
          faiss_faiss_listings: 'unavailable',
          faiss_faiss_heritage: 'unavailable'
        }
      };
    }
  }

  /**
   * Health check
   */
  async healthCheck() {
    try {
      const response = await this.request({
        method: 'GET',
        url: '/health'
      });
      return response;
    } catch (error) {
      return {
        status: 'unhealthy',
        error: error.message
      };
    }
  }
}

module.exports = new MLClient();
