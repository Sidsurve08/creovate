/**
 * AI Assist v2 Routes
 * New endpoints using ML service v2 with fallbacks
 */

const express = require('express');
const router = express.Router();
const { body, validationResult } = require('express-validator');
const { auth } = require('../middleware/auth');
const mlClient = require('../services/mlClient');

/**
 * POST /api/ai-assist-v2/listing/full-pipeline
 * Full AI-assisted listing creation pipeline with price suggestion
 */
router.post('/listing/full-pipeline', auth, [
  body('text').optional().isString(),
  body('category').optional().isString(),
  body('craft').optional().isString(),
  body('materials').optional().isArray(),
  body('technique').optional().isString(),
  body('region').optional().isString(),
  body('complexity').optional().isInt({ min: 1, max: 5 })
], async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ success: false, errors: errors.array() });
    }

    const { text, category, craft, materials, technique, region, complexity } = req.body;

    // Step 1: Classify text if provided
    let classifiedCategory = category;
    let classificationSource = 'user_input';

    if (text && !category) {
      const classification = await mlClient.classifyText(text);
      classifiedCategory = classification.category;
      classificationSource = classification.source;
    }

    // Step 2: Suggest price range
    const priceSuggestion = await mlClient.suggestPrice({
      category: classifiedCategory,
      craft,
      region,
      technique,
      complexity: complexity || 1
    });

    // Step 3: Return structured response
    res.json({
      success: true,
      data: {
        category: classifiedCategory,
        classification_source: classificationSource,
        price_range: {
          min: priceSuggestion.min_price,
          suggested: priceSuggestion.suggested_price,
          max: priceSuggestion.max_price,
          confidence: priceSuggestion.confidence,
          source: priceSuggestion.source
        },
        metadata: {
          input: { text, category, craft, materials, technique, region, complexity },
          ml_service_status: await mlClient.getModelStatus()
        }
      }
    });
  } catch (error) {
    console.error('AI assist v2 error:', error);
    res.status(500).json({
      success: false,
      message: 'AI assistance failed',
      error: error.message
    });
  }
});

/**
 * POST /api/ai-assist-v2/classify/text
 * Classify text into category
 */
router.post('/classify/text', auth, [
  body('text').notEmpty().withMessage('Text is required')
], async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ success: false, errors: errors.array() });
    }

    const { text } = req.body;
    const result = await mlClient.classifyText(text);

    res.json({
      success: true,
      data: result
    });
  } catch (error) {
    console.error('Text classification error:', error);
    res.status(500).json({
      success: false,
      message: 'Text classification failed',
      error: error.message
    });
  }
});

/**
 * POST /api/ai-assist-v2/price/suggest
 * Suggest price range for listing
 */
router.post('/price/suggest', auth, [
  body('category').optional().isString(),
  body('craft').optional().isString(),
  body('region').optional().isString(),
  body('technique').optional().isString(),
  body('complexity').optional().isInt({ min: 1, max: 5 })
], async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ success: false, errors: errors.array() });
    }

    const result = await mlClient.suggestPrice(req.body);

    res.json({
      success: true,
      data: result
    });
  } catch (error) {
    console.error('Price suggestion error:', error);
    res.status(500).json({
      success: false,
      message: 'Price suggestion failed',
      error: error.message
    });
  }
});

/**
 * GET /api/ai-assist-v2/models/status
 * Get status of ML models
 */
router.get('/models/status', async (req, res) => {
  try {
    const status = await mlClient.getModelStatus();
    res.json({
      success: true,
      data: status
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: 'Failed to get model status',
      error: error.message
    });
  }
});

module.exports = router;
