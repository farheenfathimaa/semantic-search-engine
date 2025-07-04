document.addEventListener('DOMContentLoaded', function() {
    const searchForm = document.querySelector('.search-form');
    const searchInput = document.querySelector('input[name="query"]');
    const searchButton = document.querySelector('button[type="submit"]');
    
    // Add loading state
    searchForm.addEventListener('submit', function() {
        searchButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
        searchButton.disabled = true;
    });
    
    // Add some example queries
    const exampleQueries = [
        "machine learning natural language processing",
        "deep learning computer vision",
        "neural networks attention mechanisms",
        "quantum computing algorithms",
        "blockchain healthcare applications",
        "reinforcement learning robotics"
    ];
    
    // Add placeholder rotation
    let currentExample = 0;
    if (searchInput && !searchInput.value) {
        setInterval(function() {
            searchInput.placeholder = `Try: "${exampleQueries[currentExample]}"`;
            currentExample = (currentExample + 1) % exampleQueries.length;
        }, 3000);
    }
    
    // Add smooth scrolling to results
    const resultsSection = document.querySelector('.results-section');
    if (resultsSection) {
        setTimeout(function() {
            resultsSection.scrollIntoView({ behavior: 'smooth' });
        }, 100);
    }
    
    // Add copy functionality for paper titles
    document.querySelectorAll('.paper-title').forEach(title => {
        title.addEventListener('click', function() {
            navigator.clipboard.writeText(this.textContent);
            // Show a brief feedback
            const originalText = this.textContent;
            this.textContent = '📋 Copied!';
            setTimeout(() => {
                this.textContent = originalText;
            }, 1000);
        });
    });
});