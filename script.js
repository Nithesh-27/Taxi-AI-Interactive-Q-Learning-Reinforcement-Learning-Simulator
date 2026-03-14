/**
 * Taxi AI – Q-Learning Route Optimization Simulator
 * Core Script: Handles Theme, Background, RL Logic, and UI
 */

// --- Constants & Config ---
const CONFIG = {
    gridSize: 5,
    colors: {
        taxi: '#FFD700',
        passenger: '#00E676',
        destination: '#FF5252',
        gridLine: 'rgba(255, 255, 255, 0.1)',
        accent: '#40e0d0'
    }
};

// --- DOM Elements ---
const themeToggle = document.getElementById('theme-toggle');
const bgCanvas = document.getElementById('bg-canvas');
const gridCanvas = document.getElementById('taxi-grid');
const chartCanvas = document.getElementById('performance-chart');
const startBtn = document.getElementById('start-btn');
const pauseBtn = document.getElementById('pause-btn');
const resetBtn = document.getElementById('reset-btn');
const modeToggle = document.getElementById('mode-toggle');
const speedSlider = document.getElementById('speed-slider');
const episodeDisplay = document.getElementById('episode-count');
const stepDisplay = document.getElementById('step-count');
const epsilonDisplay = document.getElementById('epsilon-value');
const rewardDisplay = document.getElementById('avg-reward');
const explanationHeader = document.getElementById('explanation-header');
const explanationPanel = document.querySelector('.collapsible');
const compareToggle = document.getElementById('compare-mode');

// --- Global State ---
let isDarkMode = true;
let isRunning = false;
let simulationSpeed = 50;
let rewardHistory = [];
let totalEpisodes = 0;
let currentRewards = 0;
let avgReward = 0;
let episodeStep = 0;

// --- Theme Management ---
function updateTheme() {
    isDarkMode = themeToggle.checked;
    document.body.className = isDarkMode ? 'dark-mode' : 'light-mode';
    initBackground();
}
themeToggle.addEventListener('change', updateTheme);

// --- Animated Background ---
const bgCtx = bgCanvas.getContext('2d');
let particles = [];

class Particle {
    constructor() { this.reset(); }
    reset() {
        this.x = Math.random() * bgCanvas.width;
        this.y = Math.random() * bgCanvas.height;
        this.size = Math.random() * 2 + 1;
        this.speedX = (Math.random() - 0.5) * 0.5;
        this.speedY = (Math.random() - 0.5) * 0.5;
        this.opacity = Math.random() * 0.5;
    }
    update() {
        this.x += this.speedX;
        this.y += this.speedY;
        if (this.x < 0 || this.x > bgCanvas.width) this.reset();
        if (this.y < 0 || this.y > bgCanvas.height) this.reset();
    }
    draw() {
        bgCtx.fillStyle = `rgba(64, 224, 208, ${this.opacity})`;
        bgCtx.beginPath();
        bgCtx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        bgCtx.fill();
    }
}

function initBackground() {
    bgCanvas.width = window.innerWidth;
    bgCanvas.height = window.innerHeight;
    particles = [];
    const count = isDarkMode ? 100 : 40;
    for (let i = 0; i < count; i++) particles.push(new Particle());
}

function animateBackground() {
    bgCtx.clearRect(0, 0, bgCanvas.width, bgCanvas.height);
    const gradient = bgCtx.createLinearGradient(0, 0, bgCanvas.width, bgCanvas.height);
    if (isDarkMode) {
        gradient.addColorStop(0, '#0b1e1d');
        gradient.addColorStop(1, '#142d2b');
    } else {
        gradient.addColorStop(0, '#f0fafa');
        gradient.addColorStop(1, '#e0f7f7');
    }
    bgCtx.fillStyle = gradient;
    bgCtx.fillRect(0, 0, bgCanvas.width, bgCanvas.height);
    if (isDarkMode) particles.forEach(p => { p.update(); p.draw(); });
    requestAnimationFrame(animateBackground);
}

// --- Taxi Environment ---
class TaxiEnv {
    constructor() {
        this.passengerLocs = [[0, 0], [0, 4], [4, 0], [4, 3]];
        this.reset();
    }
    reset() {
        this.taxiPos = { r: Math.floor(Math.random() * 5), c: Math.floor(Math.random() * 5) };
        this.passengerPosIdx = Math.floor(Math.random() * 4);
        this.destPosIdx = Math.floor(Math.random() * 4);
        while (this.destPosIdx === this.passengerPosIdx) this.destPosIdx = Math.floor(Math.random() * 4);
        this.hasPassenger = false;
        this.done = false;
        this.steps = 0;
        return this.getState();
    }
    getState() {
        const passIdx = this.hasPassenger ? 4 : this.passengerPosIdx;
        return `${this.taxiPos.r},${this.taxiPos.c},${passIdx},${this.destPosIdx}`;
    }
    step(action) {
        let reward = -1;
        this.steps++;
        if (action === 0) this.taxiPos.r = Math.max(0, this.taxiPos.r - 1); // N
        else if (action === 1) this.taxiPos.r = Math.min(4, this.taxiPos.r + 1); // S
        else if (action === 2) this.taxiPos.c = Math.min(4, this.taxiPos.c + 1); // E
        else if (action === 3) this.taxiPos.c = Math.max(0, this.taxiPos.c - 1); // W
        else if (action === 4) { // Pickup
            const pLoc = this.passengerLocs[this.passengerPosIdx];
            if (!this.hasPassenger && this.taxiPos.r === pLoc[0] && this.taxiPos.c === pLoc[1]) {
                this.hasPassenger = true;
                reward = 0;
            } else reward = -10;
        } else if (action === 5) { // Dropoff
            const dLoc = this.passengerLocs[this.destPosIdx];
            if (this.hasPassenger && this.taxiPos.r === dLoc[0] && this.taxiPos.c === dLoc[1]) {
                this.hasPassenger = false;
                this.done = true;
                reward = 20;
            } else reward = -10;
        }
        if (this.steps >= 200) this.done = true;
        return { state: this.getState(), reward, done: this.done };
    }
}

// --- Agents ---
class QAgent {
    constructor() {
        this.qTable = {};
        this.alpha = 0.2;
        this.gamma = 0.9;
        this.epsilon = 1.0;
        this.epsilonDecay = 0.9995;
        this.minEpsilon = 0.01;
        this.lastMoveExploratory = false;
    }
    getQ(state) {
        if (!this.qTable[state]) this.qTable[state] = new Array(6).fill(0);
        return this.qTable[state];
    }
    chooseAction(state, isTraining = true) {
        if (isTraining && Math.random() < this.epsilon) {
            this.lastMoveExploratory = true;
            return Math.floor(Math.random() * 6);
        }
        this.lastMoveExploratory = false;
        const values = this.getQ(state);
        // ...
        const max = Math.max(...values);
        const best = [];
        for (let i = 0; i < 6; i++) if (values[i] === max) best.push(i);
        return best[Math.floor(Math.random() * best.length)];
    }
    update(state, action, reward, nextState) {
        const currentQ = this.getQ(state)[action];
        const nextMaxQ = Math.max(...this.getQ(nextState));
        this.qTable[state][action] = currentQ + this.alpha * (reward + this.gamma * nextMaxQ - currentQ);
        if (this.epsilon > this.minEpsilon) this.epsilon *= this.epsilonDecay;
    }
}

class RandomAgent {
    constructor() { this.epsilon = 1.0; }
    chooseAction() { return Math.floor(Math.random() * 6); }
    update() { }
}

// --- Renderer ---
const env = new TaxiEnv();
const qAgent = new QAgent();
const randomAgent = new RandomAgent();
const gridCtx = gridCanvas.getContext('2d');
const chartCtx = chartCanvas.getContext('2d');

function drawGrid() {
    const isComparing = compareToggle.checked;
    const activeAgent = isComparing ? randomAgent : qAgent;
    const cellSize = gridCanvas.width / 5;
    gridCtx.clearRect(0, 0, gridCanvas.width, gridCanvas.height);

    // Heatmap & Arrows
    if (!isComparing) {
        for (let r = 0; r < 5; r++) {
            for (let c = 0; c < 5; c++) {
                const statePrefix = `${r},${c},`;
                let maxQ = -Infinity;
                Object.keys(qAgent.qTable).forEach(s => {
                    if (s.startsWith(statePrefix)) maxQ = Math.max(maxQ, ...qAgent.qTable[s]);
                });
                if (maxQ > 0) {
                    gridCtx.fillStyle = `rgba(64, 224, 208, ${Math.min(0.2, maxQ / 100)})`;
                    gridCtx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);
                }
                const state = `${r},${c},${env.hasPassenger ? 4 : env.passengerPosIdx},${env.destPosIdx}`;
                if (qAgent.qTable[state]) {
                    const values = qAgent.qTable[state];
                    const best = values.indexOf(Math.max(...values));
                    gridCtx.fillStyle = isDarkMode ? 'rgba(255, 255, 255, 0.2)' : 'rgba(0, 0, 0, 0.1)';
                    gridCtx.font = '14px Outfit';
                    gridCtx.textAlign = 'center';
                    const arrows = ['↑', '↓', '→', '←', 'P', 'D'];
                    gridCtx.fillText(arrows[best], c * cellSize + cellSize / 2, r * cellSize + cellSize / 2 + 5);
                }
            }
        }
    }

    // Grid Lines
    gridCtx.strokeStyle = isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.05)';
    for (let i = 0; i <= 5; i++) {
        gridCtx.beginPath(); gridCtx.moveTo(i * cellSize, 0); gridCtx.lineTo(i * cellSize, gridCanvas.height); gridCtx.stroke();
        gridCtx.beginPath(); gridCtx.moveTo(0, i * cellSize); gridCtx.lineTo(gridCanvas.width, i * cellSize); gridCtx.stroke();
    }

    // Destinations
    env.passengerLocs.forEach((loc, i) => {
        const x = loc[1] * cellSize + cellSize / 2, y = loc[0] * cellSize + cellSize / 2;
        gridCtx.fillStyle = isDarkMode ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.02)';
        gridCtx.beginPath(); gridCtx.arc(x, y, cellSize * 0.35, 0, Math.PI * 2); gridCtx.fill();
        gridCtx.fillStyle = isDarkMode ? '#80cbc4' : '#4a6a6a';
        gridCtx.font = 'bold 12px Outfit';
        gridCtx.fillText(['R', 'G', 'Y', 'B'][i], x, y + 4);
    });

    // Passenger & Destination
    if (!env.hasPassenger) {
        const pLoc = env.passengerLocs[env.passengerPosIdx];
        gridCtx.fillStyle = CONFIG.colors.passenger;
        gridCtx.shadowBlur = 10; gridCtx.shadowColor = CONFIG.colors.passenger;
        gridCtx.beginPath(); gridCtx.arc(pLoc[1] * cellSize + cellSize / 2, pLoc[0] * cellSize + cellSize / 2, 8, 0, Math.PI * 2); gridCtx.fill();
        gridCtx.shadowBlur = 0;
    }
    const dLoc = env.passengerLocs[env.destPosIdx];
    gridCtx.strokeStyle = CONFIG.colors.destination; gridCtx.lineWidth = 3;
    gridCtx.strokeRect(dLoc[1] * cellSize + 10, dLoc[0] * cellSize + 10, cellSize - 20, cellSize - 20);

    // Taxi
    const tx = env.taxiPos.c * cellSize + cellSize / 2, ty = env.taxiPos.r * cellSize + cellSize / 2;
    gridCtx.fillStyle = CONFIG.colors.taxi;
    gridCtx.shadowBlur = 15;
    gridCtx.shadowColor = (activeAgent.lastMoveExploratory) ? '#FF5252' : CONFIG.colors.taxi;
    gridCtx.beginPath(); gridCtx.roundRect(tx - 15, ty - 10, 30, 20, 4); gridCtx.fill();
    gridCtx.shadowBlur = 0;
    gridCtx.fillStyle = '#000'; gridCtx.fillRect(tx - 5, ty - 12, 10, 4);
    if (env.hasPassenger) {
        gridCtx.fillStyle = CONFIG.colors.passenger;
        gridCtx.beginPath(); gridCtx.arc(tx, ty, 4, 0, Math.PI * 2); gridCtx.fill();
    }
}

function drawChart() {
    const w = chartCanvas.width, h = chartCanvas.height;
    chartCtx.clearRect(0, 0, w, h);
    if (rewardHistory.length < 2) return;
    chartCtx.strokeStyle = CONFIG.colors.accent; chartCtx.lineWidth = 2;
    chartCtx.beginPath();
    rewardHistory.forEach((v, i) => {
        const x = (i / (rewardHistory.length - 1)) * w;
        const y = h - ((v + 200) / 220) * h;
        if (i === 0) chartCtx.moveTo(x, y); else chartCtx.lineTo(x, y);
    });
    chartCtx.stroke();
}

// --- Logic ---
let currentState = env.getState();
function updateStats() {
    episodeDisplay.textContent = totalEpisodes;
    stepDisplay.textContent = episodeStep;
    epsilonDisplay.textContent = qAgent.epsilon.toFixed(2);
    rewardDisplay.textContent = avgReward.toFixed(1);

    const indicator = document.getElementById('convergence-indicator');
    if (avgReward > 5) {
        indicator.textContent = "Agent Optimized";
        indicator.style.color = "#00E676";
    } else if (totalEpisodes > 100) {
        indicator.textContent = "Refining Strategy...";
        indicator.style.color = "#FFD700";
    } else {
        indicator.textContent = "Training...";
        indicator.style.color = "var(--text-secondary)";
    }
}

function runStep() {
    if (!isRunning) return;
    const isComparing = compareToggle.checked;
    const agent = isComparing ? randomAgent : qAgent;
    const isTraining = !modeToggle.checked || isComparing;

    const action = agent.chooseAction(currentState, isTraining);
    const { state: nextState, reward, done } = env.step(action);
    if (!isComparing && isTraining) qAgent.update(currentState, action, reward, nextState);

    currentState = nextState; currentRewards += reward; episodeStep++;
    drawGrid();
    if (done) {
        totalEpisodes++;
        rewardHistory.push(currentRewards);
        if (rewardHistory.length > 50) rewardHistory.shift();
        avgReward = rewardHistory.reduce((a, b) => a + b, 0) / rewardHistory.length;
        currentRewards = 0; episodeStep = 0; currentState = env.reset();
        updateStats(); drawChart();
    }
    const delay = 500 / Math.pow(speedSlider.value / 10, 2);
    setTimeout(() => { if (isRunning) requestAnimationFrame(runStep); }, delay);
}

// --- Events ---
startBtn.addEventListener('click', () => { isRunning = true; startBtn.disabled = true; pauseBtn.disabled = false; runStep(); });
pauseBtn.addEventListener('click', () => { isRunning = false; startBtn.disabled = false; pauseBtn.disabled = true; });
resetBtn.addEventListener('click', () => {
    isRunning = false; startBtn.disabled = false; pauseBtn.disabled = true;
    totalEpisodes = 0; currentRewards = 0; avgReward = 0; episodeStep = 0;
    rewardHistory = []; qAgent.qTable = {}; qAgent.epsilon = 1.0;
    currentState = env.reset(); updateStats(); drawGrid();
    chartCtx.clearRect(0, 0, chartCanvas.width, chartCanvas.height);
});

compareToggle.addEventListener('change', () => {
    isRunning = false; startBtn.disabled = false; pauseBtn.disabled = true;
    totalEpisodes = 0; currentRewards = 0; avgReward = 0; episodeStep = 0;
    rewardHistory = []; currentState = env.reset(); updateStats(); drawGrid(); drawChart();
});

document.getElementById('download-btn').addEventListener('click', () => {
    const data = {
        name: "Taxi AI Q-Learning Results",
        timestamp: new Date().toISOString(),
        total_episodes: totalEpisodes,
        avg_reward: avgReward,
        epsilon: qAgent.epsilon,
        q_table_size: Object.keys(qAgent.qTable).length,
        reward_history: rewardHistory
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `taxi-results-${Date.now()}.json`;
    a.click();
});
explanationHeader.addEventListener('click', () => explanationPanel.classList.toggle('active'));
window.addEventListener('resize', () => {
    initBackground();
    const gSize = gridCanvas.parentElement.clientWidth - 40;
    gridCanvas.width = gSize; gridCanvas.height = gSize;
    chartCanvas.width = chartCanvas.parentElement.clientWidth;
    chartCanvas.height = chartCanvas.parentElement.clientHeight;
    drawGrid(); drawChart();
});

// Start
initBackground(); animateBackground();
gridCanvas.width = gridCanvas.parentElement.clientWidth - 40;
gridCanvas.height = gridCanvas.width;
chartCanvas.width = chartCanvas.parentElement.clientWidth;
chartCanvas.height = chartCanvas.parentElement.clientHeight;
updateStats(); drawGrid();
