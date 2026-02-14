// ================================
// Tinder Automation Console Script - With Settings & Database Integration
// ================================

class TinderAutomationConsole {
    constructor() {
        this.isRunning = false;
        this.autoScroll = true;
        this.eventSource = null;
        this.statusInterval = null;
        this.currentSettings = null;
        
        // Screen mirroring
        this.socket = null;
        this.mirrorConnected = false;
        
        this.initializeElements();
        this.bindEvents();
        this.initializeSocketIO();
        this.initializeStatus();
        this.loadSettings();
        this.connectEventStream();
    }
    
    initializeElements() {
        // Control elements
        this.controlBtn = document.getElementById('controlBtn');
        this.settingsBtn = document.getElementById('settingsBtn');
        
        // Stats elements
        this.totalProfiles = document.getElementById('totalProfiles');
        this.likedProfiles = document.getElementById('likedProfiles');
        this.dislikedProfiles = document.getElementById('dislikedProfiles');
        this.likeRate = document.getElementById('likeRate');
        
        // Console elements
        this.console = document.getElementById('console');
        this.clearBtn = document.getElementById('clearBtn');
        this.scrollBtn = document.getElementById('scrollBtn');
        
        // Device screen elements
        this.mirrorStatus = document.getElementById('mirrorStatus');
        this.devicePlaceholder = document.getElementById('devicePlaceholder');
        this.deviceScreen = document.getElementById('deviceScreen');
        this.screenImage = document.getElementById('screenImage');
        
        // Modal elements
        this.errorModal = document.getElementById('errorModal');
        this.errorMessage = document.getElementById('errorMessage');
        this.successModal = document.getElementById('successModal');
        this.successMessage = document.getElementById('successMessage');
        this.settingsModal = document.getElementById('settingsModal');
        
        // Settings form elements
        this.settingsForm = document.getElementById('settingsForm');
        this.tabButtons = document.querySelectorAll('.tab-button');
        this.tabContents = document.querySelectorAll('.tab-content');
        
        // Close modal buttons
        this.closeModalButtons = document.querySelectorAll('.close-modal');
        this.closeSettingsModal = document.getElementById('closeSettingsModal');
        this.cancelSettings = document.getElementById('cancelSettings');
    }
    
    initializeSocketIO() {
        // Initialize SocketIO for screen mirroring
        this.socket = io();
        
        this.socket.on('connect', () => {
            this.updateMirrorStatus('connected', 'Connected');
            this.mirrorConnected = true;
        });
        
        this.socket.on('disconnect', () => {
            this.updateMirrorStatus('disconnected', 'Disconnected');
            this.mirrorConnected = false;
            this.hideDeviceScreen();
        });
        
        this.socket.on('frame_update', (data) => {
            this.updateDeviceScreen(data.frame);
        });
        
        this.socket.on('mirror_status', (data) => {
            if (data.status === 'started') {
                this.updateMirrorStatus('connected', 'Active');
                this.showDeviceScreen();
            } else if (data.status === 'stopped') {
                this.updateMirrorStatus('connected', 'Connected');
                this.hideDeviceScreen();
            } else if (data.status === 'error') {
                this.updateMirrorStatus('disconnected', 'Error');
                this.hideDeviceScreen();
                this.logMessage('error', `Mirror error: ${data.message}`);
            }
        });
    }
    
    bindEvents() {
        // Control buttons
        this.controlBtn.addEventListener('click', () => {
            if (this.isRunning) {
                this.stopAutomation();
            } else {
                this.startAutomation();
            }
        });
        
        this.settingsBtn.addEventListener('click', () => {
            this.showSettings();
        });
        
        // Console controls
        this.clearBtn.addEventListener('click', () => this.clearConsole());
        this.scrollBtn.addEventListener('click', () => this.toggleAutoScroll());
        
        // Settings tabs
        this.tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                const tabName = button.dataset.tab;
                this.switchTab(tabName);
            });
        });
        
        // Settings form
        this.settingsForm.addEventListener('submit', (e) => {
            e.preventDefault();
            this.saveSettings();
        });
        
        this.cancelSettings.addEventListener('click', () => {
            this.hideSettings();
        });
        
        // Modal close buttons
        this.closeModalButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                const modal = e.target.closest('.modal');
                this.hideModal(modal);
            });
        });
        
        // Click outside modal to close
        [this.errorModal, this.successModal, this.settingsModal].forEach(modal => {
            modal.addEventListener('click', (e) => {
                if (e.target === modal) {
                    this.hideModal(modal);
                }
            });
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'c') {
                if (this.isRunning) {
                    e.preventDefault();
                    this.stopAutomation();
                }
            } else if (e.key === 'Escape') {
                this.hideAllModals();
            } else if (e.ctrlKey && e.key === ',') {
                e.preventDefault();
                this.showSettings();
            }
        });
    }
    
    async initializeStatus() {
        try {
            const response = await fetch('/status');
            const data = await response.json();
            this.updateStatus(data.running);
            this.updateStats(data.stats);
        } catch (error) {
            this.logMessage('error', 'Failed to get initial status');
        }
    }
    
    async loadSettings() {
        try {
            const response = await fetch('/api/settings');
            const data = await response.json();
            
            if (data.success) {
                this.currentSettings = data.settings;
                this.populateSettingsForm(data.settings);
            } else {
                this.showError('Failed to load settings: ' + data.message);
            }
        } catch (error) {
            this.showError('Failed to load settings: ' + error.message);
        }
    }
    
    populateSettingsForm(settings) {
        // Basic settings
        document.getElementById('gemini_api_key').value = settings.gemini_api_key || '';
        document.getElementById('decision_criteria').value = settings.decision_criteria || '';
        document.getElementById('package_name').value = settings.package_name || '';
        
        // Advanced settings
        document.getElementById('model_heavy').value = settings.model_heavy || '';
        document.getElementById('model_fast').value = settings.model_fast || '';
        document.getElementById('short_wait').value = settings.short_wait || '';
        document.getElementById('medium_wait').value = settings.medium_wait || '';
        document.getElementById('long_wait').value = settings.long_wait || '';
        
        // Prompts
        document.getElementById('decision_prompt').value = settings.decision_prompt || '';
        document.getElementById('ui_resolver_prompt').value = settings.ui_resolver_prompt || '';
    }
    
    async saveSettings() {
        try {
            const formData = new FormData(this.settingsForm);
            const settings = {};
            
            for (let [key, value] of formData.entries()) {
                // Convert numeric fields
                if (['short_wait', 'medium_wait', 'long_wait'].includes(key)) {
                    settings[key] = parseFloat(value);
                } else {
                    settings[key] = value;
                }
            }
            
            // Validate required fields
            if (!settings.gemini_api_key || settings.gemini_api_key.trim() === '') {
                this.showError('Gemini API Key is required');
                return;
            }
            
            if (!settings.decision_criteria || settings.decision_criteria.trim() === '') {
                this.showError('Decision Criteria is required');
                return;
            }
            
            // Show loading state
            const submitBtn = this.settingsForm.querySelector('button[type="submit"]');
            const originalText = submitBtn.textContent;
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Saving...';
            
            const response = await fetch('/api/settings', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(settings)
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.currentSettings = data.settings;
                this.hideSettings();
                this.showSuccess('Settings saved successfully');
                this.logMessage('success', 'Settings updated');
            } else {
                this.showError('Failed to save settings: ' + data.message);
            }
        } catch (error) {
            this.showError('Failed to save settings: ' + error.message);
        } finally {
            // Reset button state
            const submitBtn = this.settingsForm.querySelector('button[type="submit"]');
            submitBtn.disabled = false;
            submitBtn.textContent = 'Save Settings';
        }
    }
    
    switchTab(tabName) {
        // Update tab buttons
        this.tabButtons.forEach(button => {
            button.classList.remove('active');
            if (button.dataset.tab === tabName) {
                button.classList.add('active');
            }
        });
        
        // Update tab content
        this.tabContents.forEach(content => {
            content.style.display = 'none';
            if (content.id === `${tabName}-tab`) {
                content.style.display = 'block';
            }
        });
    }
    
    showSettings() {
        this.loadSettings();
        this.showModal(this.settingsModal);
    }
    
    hideSettings() {
        this.hideModal(this.settingsModal);
    }
    
    showModal(modal) {
        modal.classList.add('show');
    }
    
    hideModal(modal) {
        modal.classList.remove('show');
    }
    
    hideAllModals() {
        [this.errorModal, this.successModal, this.settingsModal].forEach(modal => {
            this.hideModal(modal);
        });
    }
    
    connectEventStream() {
        // Close existing connection
        if (this.eventSource) {
            this.eventSource.close();
        }
        
        // Create new EventSource for real-time logs
        this.eventSource = new EventSource('/logs');
        
        this.eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                
                if (data.type === 'heartbeat') {
                    return; // Ignore heartbeat messages
                }
                
                this.handleLogMessage(data);
            } catch (error) {
                console.error('Error parsing event data:', error);
            }
        };
        
        this.eventSource.onerror = (error) => {
            console.error('EventSource error:', error);
            
            // Attempt to reconnect after 5 seconds
            setTimeout(() => {
                if (this.eventSource.readyState === EventSource.CLOSED) {
                    this.connectEventStream();
                }
            }, 5000);
        };
        
        // Start status polling
        this.startStatusPolling();
    }
    
    startStatusPolling() {
        // Poll status every 2 seconds
        this.statusInterval = setInterval(async () => {
            try {
                const response = await fetch('/status');
                const data = await response.json();
                this.updateStatus(data.running);
                this.updateStats(data.stats);
            } catch (error) {
                // Silently fail status updates
            }
        }, 2000);
    }
    
    async startAutomation() {
        // Check if settings are configured
        if (!this.currentSettings || !this.currentSettings.gemini_api_key) {
            this.showError('Please configure settings before starting automation');
            this.showSettings();
            return;
        }
        
        this.setButtonLoading(true);
        
        try {
            const response = await fetch('/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.updateStatus(true);
                this.logMessage('success', data.message);
            } else {
                this.showError(data.message);
                this.logMessage('error', data.message);
            }
        } catch (error) {
            const message = 'Failed to start automation: ' + error.message;
            this.showError(message);
            this.logMessage('error', message);
        }
        
        this.setButtonLoading(false);
    }
    
    async stopAutomation() {
        this.setButtonLoading(true);
        
        try {
            const response = await fetch('/stop', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.updateStatus(false);
                this.logMessage('success', data.message);
            } else {
                this.showError(data.message);
                this.logMessage('error', data.message);
            }
        } catch (error) {
            const message = 'Failed to stop automation: ' + error.message;
            this.showError(message);
            this.logMessage('error', message);
        }
        
        this.setButtonLoading(false);
    }
    
    updateStatus(running) {
        this.isRunning = running;
        
        if (running) {
            this.controlBtn.className = 'header-control-button stop-button';
            this.controlBtn.innerHTML = '<i class="fas fa-stop"></i>';
            this.controlBtn.title = 'Stop Automation';
            this.settingsBtn.disabled = true;
            this.settingsBtn.title = 'Settings disabled during automation';
        } else {
            this.controlBtn.className = 'header-control-button start-button';
            this.controlBtn.innerHTML = '<i class="fas fa-play"></i>';
            this.controlBtn.title = 'Start Automation';
            this.settingsBtn.disabled = false;
            this.settingsBtn.title = 'Settings';
        }
        
        this.controlBtn.disabled = false;
    }
    
    updateStats(stats) {
        this.totalProfiles.textContent = stats.total || 0;
        this.likedProfiles.textContent = stats.liked || 0;
        this.dislikedProfiles.textContent = stats.disliked || 0;
        
        const total = stats.total || 0;
        const liked = stats.liked || 0;
        const rate = total > 0 ? Math.round((liked / total) * 100) : 0;
        this.likeRate.textContent = rate + '%';
    }
    
    updateMirrorStatus(status, message) {
        this.mirrorStatus.className = `mirror-status ${status}`;
        this.mirrorStatus.querySelector('span').textContent = message;
    }
    
    showDeviceScreen() {
        this.devicePlaceholder.style.display = 'none';
        this.deviceScreen.style.display = 'flex';
    }
    
    hideDeviceScreen() {
        this.devicePlaceholder.style.display = 'block';
        this.deviceScreen.style.display = 'none';
    }
    
    updateDeviceScreen(frameData) {
        if (this.screenImage && frameData) {
            this.screenImage.src = frameData;
            this.showDeviceScreen();
        }
    }
    
    handleLogMessage(data) {
        const { type, message, timestamp } = data;
        
        switch (type) {
            case 'print':
            case 'log':
                this.logMessage('info', message, timestamp);
                break;
            case 'status':
                this.logMessage('success', message, timestamp);
                break;
            case 'error':
                this.logMessage('error', message, timestamp);
                break;
            default:
                this.logMessage('info', message, timestamp);
        }
    }
    
    logMessage(type, message, timestamp) {
        const logLine = document.createElement('div');
        logLine.className = `console-line ${type}`;
        
        const timeSpan = document.createElement('span');
        timeSpan.className = 'timestamp';
        timeSpan.textContent = timestamp || this.getCurrentTime();
        
        const messageSpan = document.createElement('span');
        messageSpan.className = 'message';
        messageSpan.textContent = message;
        
        logLine.appendChild(timeSpan);
        logLine.appendChild(messageSpan);
        
        this.console.appendChild(logLine);
        
        // Auto-scroll if enabled
        if (this.autoScroll) {
            this.console.scrollTop = this.console.scrollHeight;
        }
        
        // Limit console lines (keep last 1000 lines)
        const lines = this.console.children;
        if (lines.length > 1000) {
            this.console.removeChild(lines[0]);
        }
    }
    
    getCurrentTime() {
        const now = new Date();
        return now.toTimeString().substr(0, 8);
    }
    
    clearConsole() {
        this.console.innerHTML = '<div class="console-line welcome"><span class="timestamp">[System]</span><span class="message">Console cleared</span></div>';
    }
    
    toggleAutoScroll() {
        this.autoScroll = !this.autoScroll;
        
        if (this.autoScroll) {
            this.scrollBtn.classList.add('active');
            this.scrollBtn.title = 'Auto-scroll: On';
            this.console.scrollTop = this.console.scrollHeight;
        } else {
            this.scrollBtn.classList.remove('active');
            this.scrollBtn.title = 'Auto-scroll: Off';
        }
    }
    
    setButtonLoading(loading) {
        if (loading) {
            this.controlBtn.disabled = true;
            this.controlBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
            this.controlBtn.title = 'Processing...';
        } else {
            this.controlBtn.disabled = false;
        }
    }
    
    showError(message) {
        this.errorMessage.textContent = message;
        this.showModal(this.errorModal);
        
        // Auto-hide after 5 seconds
        setTimeout(() => {
            this.hideModal(this.errorModal);
        }, 5000);
    }
    
    showSuccess(message) {
        this.successMessage.textContent = message;
        this.showModal(this.successModal);
        
        // Auto-hide after 3 seconds
        setTimeout(() => {
            this.hideModal(this.successModal);
        }, 3000);
    }
    
    destroy() {
        // Clean up resources
        if (this.eventSource) {
            this.eventSource.close();
        }
        
        if (this.statusInterval) {
            clearInterval(this.statusInterval);
        }
        
        if (this.socket) {
            this.socket.disconnect();
        }
    }
}

// Initialize the console when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.tinderConsole = new TinderAutomationConsole();
});

// Clean up when the page is unloaded
window.addEventListener('beforeunload', () => {
    if (window.tinderConsole) {
        window.tinderConsole.destroy();
    }
});