/**
 * Next Move AI — Mathematical Rendering Module (KaTeX + Auto-Render)
 * Configures inline \( ... \) and display \[ ... \] mathematics.
 * Handles both initial static HTML and dynamically updated elements.
 */

const MathRenderer = {
    isReady: false,
    options: {
        delimiters: [
            { left: '$$', right: '$$', display: true },
            { left: '\\[', right: '\\]', display: true },
            { left: '\\(', right: '\\)', display: false },
            { left: '$', right: '$', display: false }
        ],
        throwOnError: false,
        errorColor: '#f43f5e',
        ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
    },

    init() {
        const render = () => {
            if (window.renderMathInElement) {
                this.isReady = true;
                window.renderMathInElement(document.body, this.options);
            }
        };

        if (window.renderMathInElement) {
            render();
        } else {
            // KaTeX scripts are loaded with defer
            window.addEventListener('load', render);
        }
    },

    renderElement(element) {
        if (!element) return;
        if (window.renderMathInElement) {
            window.renderMathInElement(element, this.options);
        } else if (!this.isReady) {
            window.addEventListener('load', () => {
                if (window.renderMathInElement) {
                    window.renderMathInElement(element, this.options);
                }
            }, { once: true });
        }
    }
};

window.MathRenderer = MathRenderer;
window.renderMath = (el = document.body) => MathRenderer.renderElement(el);

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => MathRenderer.init());
