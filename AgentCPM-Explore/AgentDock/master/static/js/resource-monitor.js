// Resource Monitor JavaScript Module
class ResourceMonitor {
    constructor() {
        this.autoRefreshInterval = null;
        this.refreshRate = 5000; // 5 seconds
        this.charts = {};
        this.lastUpdateTime = null;
        this.isUpdating = false;
    }

    // Initialize monitor
    async initialize() {
        console.log('Initializing resource monitor...');
        await this.refreshResources();
        this.setupEventListeners();
        this.initializeCharts();
    }

    // Setup event listeners
    setupEventListeners() {
        // Handle page visibility changes
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.pauseAutoRefresh();
            } else {
                if (this.autoRefreshInterval) {
                    this.resumeAutoRefresh();
                }
            }
        });

        // Handle window focus changes
        window.addEventListener('focus', () => {
            this.refreshResources();
        });
    }

    // Refresh all resource info
    async refreshResources() {
        if (this.isUpdating) return;
        
        this.isUpdating = true;
        this.showLoadingIndicator();
        
        try {
            await Promise.all([
                this.updateSystemResources(),
                this.updateDockerResources(),
                this.updateAlerts()
            ]);
            this.lastUpdateTime = new Date();
            this.updateLastRefreshTime();
        } catch (error) {
            console.error('Failed to refresh resources:', error);
            this.showError('Refresh failed: ' + error.message);
        } finally {
            this.isUpdating = false;
            this.hideLoadingIndicator();
        }
    }

    // Show loading indicator
    showLoadingIndicator() {
        const refreshIcon = document.getElementById('refreshIcon');
        if (refreshIcon) {
            refreshIcon.classList.add('refresh-indicator');
        }
    }

    // Hide loading indicator
    hideLoadingIndicator() {
        const refreshIcon = document.getElementById('refreshIcon');
        if (refreshIcon) {
            refreshIcon.classList.remove('refresh-indicator');
        }
    }

    // Update system resources
    async updateSystemResources() {
        try {
            const response = await fetch('/api/resources/system');
            const result = await response.json();
            
            if (result.status === 'success') {
                const data = result.data;
                this.renderSystemMetrics(data);
                this.updateCharts(data);
            } else {
                throw new Error(result.message || 'Failed to get system resources');
            }
        } catch (error) {
            console.error('Failed to update system resources:', error);
            throw error;
        }
    }

    // Render system metrics
    renderSystemMetrics(data) {
        // CPU
        this.updateMetricWithAnimation('cpuUsage', data.cpu.percent + '%');
        this.updateProgressBar('cpuProgress', data.cpu.percent, this.getCpuColor(data.cpu.percent));
        this.updateElement('cpuCores', data.cpu.count + ' cores');
        
        // Memory
        this.updateMetricWithAnimation('memoryUsage', data.memory.percent + '%');
        this.updateProgressBar('memoryProgress', data.memory.percent, this.getMemoryColor(data.memory.percent));
        this.updateElement('memoryDetails', `${data.memory.used_gb}GB / ${data.memory.total_gb}GB`);
        
        // Load
        this.updateMetricWithAnimation('loadAverage', data.load_average['1min']);
        this.updateElement('loadDetails', `5min: ${data.load_average['5min']} | 15min: ${data.load_average['15min']}`);
        
        // Disk
        this.updateMetricWithAnimation('diskUsage', data.disk.percent + '%');
        this.updateProgressBar('diskProgress', data.disk.percent, '#fd7e14');
        this.updateElement('diskDetails', `${data.disk.used_gb}GB / ${data.disk.total_gb}GB`);
    }

    // Update metric with animation
    updateMetricWithAnimation(elementId, value) {
        const element = document.getElementById(elementId);
        if (element) {
            element.classList.add('updating');
            element.textContent = value;
            setTimeout(() => {
                element.classList.remove('updating');
            }, 300);
        }
    }

    // Update progress bar
    updateProgressBar(elementId, percentage, color) {
        const element = document.getElementById(elementId);
        if (element) {
            element.style.width = percentage + '%';
            element.style.backgroundColor = color;
        }
    }

    // Update element content
    updateElement(elementId, content) {
        const element = document.getElementById(elementId);
        if (element) {
            element.textContent = content;
        }
    }

    // Get CPU color
    getCpuColor(percentage) {
        if (percentage > 80) return '#dc3545';
        if (percentage > 60) return '#ffc107';
        return '#17a2b8';
    }

    // Get memory color
    getMemoryColor(percentage) {
        if (percentage > 85) return '#dc3545';
        if (percentage > 70) return '#ffc107';
        return '#28a745';
    }

    // Update Docker container info
    async updateDockerResources() {
        try {
            const response = await fetch('/api/resources/docker');
            const result = await response.json();
            
            if (result.status === 'success') {
                const data = result.data;
                this.renderDockerMetrics(data);
            } else {
                throw new Error(result.message || 'Failed to get Docker resources');
            }
        } catch (error) {
            console.error('Failed to update Docker containers:', error);
            throw error;
        }
    }

    // Render Docker metrics
    renderDockerMetrics(data) {
        // Update statistics
        this.updateElement('containerCount', data.summary.total_containers);
        this.updateElement('limitedCount', data.summary.limited_containers);
        this.updateElement('unlimitedCount', data.summary.unlimited_containers);
        
        // Render container list
        this.renderContainerList(data.containers);
    }

    // Render container list
    renderContainerList(containers) {
        const containersList = document.getElementById('containersList');
        if (!containersList) return;
        
        if (containers.length === 0) {
            containersList.innerHTML = `
                <div class="text-center text-muted">
                    <i class="fas fa-cube fa-2x"></i>
                    <p class="mt-2">No running containers</p>
                </div>
            `;
            return;
        }

        containersList.innerHTML = containers.map(container => {
            return this.renderContainerRow(container);
        }).join('');
    }

    // Render single container row
    renderContainerRow(container) {
        const hasLimits = container.limits && (container.limits.has_memory_limit || container.limits.has_cpu_limit);
        const limitBadge = hasLimits 
            ? '<span class="badge bg-success ms-2">Limited</span>' 
            : '<span class="badge bg-warning ms-2">Unlimited</span>';
        
        const cpuPercent = container.cpu ? container.cpu.percent.toFixed(1) : '0';
        const memoryMb = container.memory ? container.memory.usage_mb : '0';
        const cpuLimit = container.limits && container.limits.cpu_limit_cores ? 
            `${container.limits.cpu_limit_cores} cores` : 'No CPU limit';
        const memoryLimit = container.limits && container.limits.memory_limit_gb ? 
            `${container.limits.memory_limit_gb}GB` : 'No memory limit';
        
        return `
            <div class="container-row">
                <div class="d-flex justify-content-between align-items-center">
                    <div class="flex-grow-1">
                        <div class="d-flex align-items-center">
                            <span class="status-indicator status-running"></span>
                            <strong>${container.name}</strong>
                            ${limitBadge}
                        </div>
                        <small class="text-light">${container.image}</small>
                    </div>
                    <div class="text-end">
                        <div class="d-flex gap-3">
                            <div class="text-center">
                                <div class="fw-bold">${cpuPercent}%</div>
                                <small>CPU</small>
                            </div>
                            <div class="text-center">
                                <div class="fw-bold">${memoryMb}MB</div>
                                <small>Memory</small>
                            </div>
                            <div class="text-center">
                                <small>Limits:</small>
                                <div style="font-size: 0.8rem;">
                                    ${cpuLimit}<br>
                                    ${memoryLimit}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    // Update alerts
    async updateAlerts() {
        try {
            const response = await fetch('/api/resources/alerts');
            const result = await response.json();
            
            if (result.status === 'success') {
                const data = result.data;
                this.renderAlerts(data);
            }
        } catch (error) {
            console.error('Failed to update alerts:', error);
        }
    }

    // Render alerts
    renderAlerts(data) {
        // Update alert badge
        const alertBadge = document.getElementById('alertBadge');
        if (alertBadge) {
            if (data.alert_count > 0) {
                alertBadge.textContent = data.alert_count;
                alertBadge.style.display = 'flex';
                alertBadge.style.backgroundColor = data.has_critical ? '#dc3545' : '#ffc107';
            } else {
                alertBadge.style.display = 'none';
            }
        }
    }

    // Initialize charts
    initializeCharts() {
        console.log('Initializing charts...');
    }

    // Update charts
    updateCharts(data) {
        if (this.charts.memoryChart) {
            // Update memory chart
        }
    }

    // Toggle auto refresh
    toggleAutoRefresh() {
        const icon = document.getElementById('autoRefreshIcon');
        const text = document.getElementById('autoRefreshText');
        
        if (this.autoRefreshInterval) {
            this.pauseAutoRefresh();
            if (icon) icon.className = 'fas fa-play';
            if (text) text.textContent = 'Enable Auto Refresh';
        } else {
            this.startAutoRefresh();
            if (icon) icon.className = 'fas fa-pause';
            if (text) text.textContent = 'Disable Auto Refresh';
        }
    }

    // Start auto refresh
    startAutoRefresh() {
        this.autoRefreshInterval = setInterval(() => {
            this.refreshResources();
        }, this.refreshRate);
    }

    // Pause auto refresh
    pauseAutoRefresh() {
        if (this.autoRefreshInterval) {
            clearInterval(this.autoRefreshInterval);
            this.autoRefreshInterval = null;
        }
    }

    // Resume auto refresh
    resumeAutoRefresh() {
        if (!this.autoRefreshInterval) {
            this.startAutoRefresh();
        }
    }

    // Emergency stop containers
    async emergencyStop() {
        if (!confirm('Are you sure you want to emergency stop high-usage Docker containers? This will stop the top 3 resource-consuming non-critical containers.')) {
            return;
        }
        
        try {
            const response = await fetch('/api/resources/emergency-stop', {
                method: 'POST'
            });
            const result = await response.json();
            
            if (result.status === 'success') {
                const stoppedContainers = result.data.stopped_containers
                    .map(c => `- ${c.name} (CPU: ${c.cpu_percent}%)`)
                    .join('\n');
                
                alert(`${result.message}\nStopped containers:\n${stoppedContainers}`);
                await this.refreshResources();
            } else {
                alert('Emergency stop failed: ' + result.message);
            }
        } catch (error) {
            console.error('Emergency stop failed:', error);
            alert('Emergency stop failed: ' + error.message);
        }
    }

    // Show error message
    showError(message) {
        console.error(message);
    }

    // Update last refresh time
    updateLastRefreshTime() {
        const timeElement = document.getElementById('lastUpdateTime');
        if (timeElement && this.lastUpdateTime) {
            timeElement.textContent = this.lastUpdateTime.toLocaleTimeString();
        }
    }
}

// Global resource monitor instance
let resourceMonitor = null;

// Initialize after page load
document.addEventListener('DOMContentLoaded', async () => {
    resourceMonitor = new ResourceMonitor();
    await resourceMonitor.initialize();
});

// Export global functions for HTML use
window.toggleAutoRefresh = () => {
    if (resourceMonitor) {
        resourceMonitor.toggleAutoRefresh();
    }
};

window.refreshResources = () => {
    if (resourceMonitor) {
        resourceMonitor.refreshResources();
    }
};

window.emergencyStop = () => {
    if (resourceMonitor) {
        resourceMonitor.emergencyStop();
    }
};
