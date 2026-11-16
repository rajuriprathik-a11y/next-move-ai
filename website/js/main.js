/**
 * Next Move AI — Main Application Controller
 * Handles navigation, interactive tabs, ablation selector, and scroll interactions.
 */

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initAblationSelector();
    initTabControllers();
});

// Sticky Navigation and ScrollSpy
function initNavigation() {
    const navbar = document.querySelector('.navbar');
    const navLinks = document.querySelectorAll('.nav-links a');
    const sections = document.querySelectorAll('section[id]');
    const mobileToggle = document.querySelector('.mobile-nav-toggle');
    const navList = document.querySelector('.nav-links');

    // Sticky header shadow
    window.addEventListener('scroll', () => {
        if (window.scrollY > 30) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }

        // ScrollSpy
        let currentSectionId = '';
        const scrollY = window.pageYOffset;

        sections.forEach(section => {
            const sectionHeight = section.offsetHeight;
            const sectionTop = section.offsetTop - 120;
            const sectionId = section.getAttribute('id');

            if (scrollY > sectionTop && scrollY <= sectionTop + sectionHeight) {
                currentSectionId = sectionId;
            }
        });

        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === `#${currentSectionId}`) {
                link.classList.add('active');
            }
        });
    });

    // Mobile nav toggle
    if (mobileToggle && navList) {
        mobileToggle.addEventListener('click', () => {
            const expanded = mobileToggle.getAttribute('aria-expanded') === 'true';
            mobileToggle.setAttribute('aria-expanded', !expanded);
            navList.classList.toggle('open');
        });

        // Close menu on link click
        navLinks.forEach(link => {
            link.addEventListener('click', () => {
                navList.classList.remove('open');
                mobileToggle.setAttribute('aria-expanded', 'false');
            });
        });
    }
}

// Interactive Ablation Study Selector
function initAblationSelector() {
    const rows = document.querySelectorAll('.ablation-table tbody tr');
    const summaryCard = document.getElementById('ablation-selected-summary');
    if (!rows.length || !summaryCard) return;

    rows.forEach(row => {
        row.addEventListener('click', () => {
            rows.forEach(r => r.classList.remove('active-row'));
            row.classList.add('active-row');

            const name = row.dataset.name;
            const auc = row.dataset.auc;
            const delta = row.dataset.delta;
            const logloss = row.dataset.logloss;
            const role = row.dataset.role;

            summaryCard.innerHTML = `
                <h4>${name}</h4>
                <div class="ablation-stat-grid">
                    <div><span class="muted">ROC-AUC:</span> <strong>${auc}</strong></div>
                    <div><span class="muted">Δ AUC vs Full:</span> <strong class="${parseFloat(delta) < 0 ? 'text-negative' : 'text-positive'}">${delta}</strong></div>
                    <div><span class="muted">Log Loss:</span> <strong>${logloss}</strong></div>
                </div>
                <p class="ablation-role-desc">${role}</p>
            `;
        });
    });
}

// Interactive Tab Controllers (e.g. Simulator vs Model Explorer)
function initTabControllers() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabPanels = document.querySelectorAll('.tab-panel');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetId = btn.getAttribute('data-tab');

            tabBtns.forEach(b => {
                b.classList.remove('active');
                b.setAttribute('aria-selected', 'false');
            });
            tabPanels.forEach(p => p.classList.remove('active'));

            btn.classList.add('active');
            btn.setAttribute('aria-selected', 'true');

            const targetPanel = document.getElementById(targetId);
            if (targetPanel) {
                targetPanel.classList.add('active');
            }
        });
    });
}
