/**
 * Next Move AI — Theme Management (Light, Dark, System)
 * Persists user preference in localStorage and handles system theme changes.
 */

const THEME_KEY = 'next_move_ai_theme';

const ThemeManager = {
    currentTheme: 'system', // 'light' | 'dark' | 'system'

    init() {
        const saved = localStorage.getItem(THEME_KEY);
        if (saved && ['light', 'dark', 'system'].includes(saved)) {
            this.currentTheme = saved;
        } else {
            this.currentTheme = 'system';
        }

        this.applyTheme(this.currentTheme, false);

        // Listen for OS theme changes
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', e => {
            if (this.currentTheme === 'system') {
                this.applyTheme('system', false);
            }
        });

        this.bindUI();
    },

    getResolvedTheme() {
        if (this.currentTheme === 'system') {
            return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
        }
        return this.currentTheme;
    },

    applyTheme(theme, save = true) {
        this.currentTheme = theme;
        if (save) {
            localStorage.setItem(THEME_KEY, theme);
        }

        const resolved = this.getResolvedTheme();
        document.documentElement.setAttribute('data-theme', resolved);
        document.documentElement.setAttribute('data-theme-setting', theme);

        // Update UI buttons if present
        document.querySelectorAll('.theme-btn').forEach(btn => {
            const btnTheme = btn.getAttribute('data-theme-val');
            if (btnTheme === theme) {
                btn.classList.add('active');
                btn.setAttribute('aria-pressed', 'true');
            } else {
                btn.classList.remove('active');
                btn.setAttribute('aria-pressed', 'false');
            }
        });

        // Dispatch event for charts and interactive elements
        window.dispatchEvent(new CustomEvent('themeChanged', { detail: { theme: resolved, setting: theme } }));
    },

    bindUI() {
        document.querySelectorAll('.theme-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const val = btn.getAttribute('data-theme-val');
                this.applyTheme(val, true);
            });
        });
    }
};

window.ThemeManager = ThemeManager;
document.addEventListener('DOMContentLoaded', () => ThemeManager.init());
