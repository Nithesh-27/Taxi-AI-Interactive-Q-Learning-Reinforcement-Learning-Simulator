/**
 * Headless verification script for Taxi AI Q-Learning Logic
 */

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

class QAgent {
    constructor() {
        this.qTable = {};
        this.alpha = 0.2;
        this.gamma = 0.9;
        this.epsilon = 1.0;
        this.epsilonDecay = 0.9995;
        this.minEpsilon = 0.01;
    }
    getQ(state) {
        if (!this.qTable[state]) this.qTable[state] = new Array(6).fill(0);
        return this.qTable[state];
    }
    chooseAction(state, isTraining = true) {
        if (isTraining && Math.random() < this.epsilon) return Math.floor(Math.random() * 6);
        const values = this.getQ(state);
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

const env = new TaxiEnv();
const agent = new QAgent();

console.log("Starting Taxi AI Headless Simulation...");
console.log("---------------------------------------");

let rewards = [];
for (let episode = 1; episode <= 500; episode++) {
    let state = env.reset();
    let totalReward = 0;
    let done = false;

    while (!done) {
        const action = agent.chooseAction(state, true);
        const result = env.step(action);
        agent.update(state, action, result.reward, result.state);
        state = result.state;
        totalReward += result.reward;
        done = result.done;
    }

    rewards.push(totalReward);

    if (episode % 100 === 0) {
        const avg = rewards.slice(-100).reduce((a, b) => a + b, 0) / 100;
        console.log(`Episode ${episode}: Avg Reward (last 100) = ${avg.toFixed(2)}, Epsilon = ${agent.epsilon.toFixed(3)}`);
    }
}

console.log("---------------------------------------");
console.log("Simulation Complete.");
if (rewards.slice(-50).reduce((a, b) => a + b, 0) / 50 > 0) {
    console.log("Result: Agent successfully converged (positive average reward)!");
} else {
    console.log("Result: Agent is still learning.");
}
