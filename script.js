/**
 * Iris Flower Classifier - Frontend Interaction Controller
 * Manages slider-input synchronization, AJAX predictions, and dynamic UI updates.
 */

document.addEventListener('DOMContentLoaded', () => {
  // Elements configuration
  const controls = [
    {
      id: 'sepal-length',
      slider: document.getElementById('sepal-length-slider'),
      number: document.getElementById('sepal-length-num'),
      min: 4.0,
      max: 8.0,
      param: 'sepal_length'
    },
    {
      id: 'sepal-width',
      slider: document.getElementById('sepal-width-slider'),
      number: document.getElementById('sepal-width-num'),
      min: 2.0,
      max: 4.5,
      param: 'sepal_width'
    },
    {
      id: 'petal-length',
      slider: document.getElementById('petal-length-slider'),
      number: document.getElementById('petal-length-num'),
      min: 1.0,
      max: 7.0,
      param: 'petal_length'
    },
    {
      id: 'petal-width',
      slider: document.getElementById('petal-width-slider'),
      number: document.getElementById('petal-width-num'),
      min: 0.1,
      max: 2.5,
      param: 'petal_width'
    }
  ];

  const predictBtn = document.getElementById('predict-btn');
  const errorBanner = document.getElementById('form-error');

  // Result display elements
  const speciesNameEl = document.getElementById('result-species-name');
  const confidenceValEl = document.getElementById('result-confidence-val');
  const flowerImgEl = document.getElementById('result-flower-img');

  const probBars = {
    'Iris-setosa': {
      bar: document.getElementById('bar-setosa'),
      label: document.getElementById('pct-setosa')
    },
    'Iris-versicolor': {
      bar: document.getElementById('bar-versicolor'),
      label: document.getElementById('pct-versicolor')
    },
    'Iris-virginica': {
      bar: document.getElementById('bar-virginica'),
      label: document.getElementById('pct-virginica')
    }
  };

  /**
   * Update the filled track color of range sliders
   */
  function updateSliderFill(slider, min, max) {
    const val = parseFloat(slider.value);
    const percentage = ((val - min) / (max - min)) * 100;
    slider.style.background = `linear-gradient(to right, #5e48d8 0%, #5e48d8 ${percentage}%, #dfe3ec ${percentage}%, #dfe3ec 100%)`;
  }

  // Bind synchronization events for each control
  controls.forEach(({ slider, number, min, max }) => {
    // Initialize slider visual fill
    updateSliderFill(slider, min, max);

    // Slider -> Number input
    slider.addEventListener('input', () => {
      number.value = parseFloat(slider.value).toFixed(1);
      updateSliderFill(slider, min, max);
      clearError();
    });

    // Number input -> Slider
    number.addEventListener('change', () => {
      let val = parseFloat(number.value);
      if (isNaN(val)) val = min;
      // Clamp within limits
      if (val < min) val = min;
      if (val > max) val = max;
      number.value = val.toFixed(1);
      slider.value = val;
      updateSliderFill(slider, min, max);
      clearError();
    });

    number.addEventListener('input', () => {
      clearError();
    });
  });

  function showError(message) {
    errorBanner.textContent = message;
    errorBanner.style.display = 'block';
  }

  function clearError() {
    errorBanner.textContent = '';
    errorBanner.style.display = 'none';
  }

  /**
   * Perform species prediction via backend API
   */
  async function performPrediction() {
    clearError();

    // Prepare payload
    const payload = {};
    for (const ctrl of controls) {
      const val = parseFloat(ctrl.number.value);
      if (isNaN(val)) {
        showError(`Please enter a valid number for ${ctrl.id.replace('-', ' ')}.`);
        return;
      }
      payload[ctrl.param] = val;
    }

    // Set loading UI state
    predictBtn.classList.add('loading');
    predictBtn.disabled = true;

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify(payload)
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(data.error || 'Failed to predict species.');
      }

      // Update UI with results
      renderResults(data);

    } catch (err) {
      console.error('Prediction failed:', err);
      showError(err.message || 'Unable to connect to the prediction server.');
    } finally {
      predictBtn.classList.remove('loading');
      predictBtn.disabled = false;
    }
  }

  function formatPercentage(val) {
    if (val > 0 && val < 0.1) {
      return `${val.toFixed(2)}%`;
    }
    if (val >= 99.9 && val < 100) {
      return `${val.toFixed(2)}%`;
    }
    return `${val.toFixed(1)}%`;
  }

  /**
   * Render returned prediction results on the page
   */
  function renderResults(data) {
    const predictedSpecies = data.prediction;
    const confidence = data.confidence;
    const probabilities = data.probabilities;

    // Update Species Name
    speciesNameEl.textContent = predictedSpecies;

    // Update Confidence Score
    confidenceValEl.textContent = formatPercentage(confidence);

    // Update Image with gentle fade transition
    if (data.image_url) {
      flowerImgEl.style.opacity = '0.4';
      setTimeout(() => {
        flowerImgEl.src = data.image_url;
        flowerImgEl.alt = `${predictedSpecies} flower`;
        flowerImgEl.style.opacity = '1';
      }, 150);
    }

    // Update all 3 horizontal probability bars
    for (const [species, barGroup] of Object.entries(probBars)) {
      if (!barGroup || !barGroup.bar) continue;
      
      const probValue = probabilities[species] !== undefined ? probabilities[species] : 0.0;
      const formattedPct = formatPercentage(probValue);

      barGroup.label.textContent = formattedPct;
      barGroup.bar.style.width = `${Math.max(probValue, 0.5)}%`;
      barGroup.bar.closest('.progress-track')?.setAttribute('aria-valuenow', probValue);

      // Highlight predicted species bar in green, others in purple
      if (species === predictedSpecies) {
        barGroup.bar.className = 'progress-fill fill-green';
      } else {
        barGroup.bar.className = 'progress-fill fill-purple';
      }
    }
  }

  // Predict button event listener
  predictBtn.addEventListener('click', performPrediction);

  // Optional: Allow pressing Enter inside number inputs to predict
  controls.forEach(({ number }) => {
    number.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        performPrediction();
      }
    });
  });

  // Execute initial prediction on page load to initialize with exact model outputs
  performPrediction();
});
