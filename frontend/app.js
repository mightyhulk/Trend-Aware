// ══════════════════════════════════════════════
// AI Trend-Aware Recommendation System — Frontend
// ══════════════════════════════════════════════

const API = 'http://localhost:8000/api';

// ─── Tab Navigation ───
document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById(`tab-${btn.dataset.tab}`).classList.add('active');
    });
});

// ─── Slider labels ───
document.getElementById('trend-weight').addEventListener('input', e => {
    document.getElementById('tw-val').textContent = e.target.value;
});
document.getElementById('diversity').addEventListener('input', e => {
    document.getElementById('div-val').textContent = e.target.value;
});

// ─── API Helpers ───
async function api(path) {
    const r = await fetch(`${API}${path}`);
    return r.json();
}
async function apiPost(path, body) {
    const r = await fetch(`${API}${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
    return r.json();
}

function loading(el) {
    el.innerHTML = '<div class="loading"><div class="spinner"></div>Loading...</div>';
}

// ─── Dashboard ───
async function loadDashboard() {
    const metricsEl = document.getElementById('metrics-grid');
    const trendingEl = document.getElementById('dashboard-trending');
    const coldEl = document.getElementById('cold-start-info');

    loading(metricsEl); loading(trendingEl); loading(coldEl);

    const [metrics, trending, coldStart] = await Promise.all([
        api('/system/metrics'),
        api('/trending?top_n=8'),
        api('/cold-start/status'),
    ]);

    const ds = metrics.data_statistics;
    const cs = metrics.cold_start;
    const tr = metrics.training_metrics;
    const cf = tr?.collaborative || {};

    metricsEl.innerHTML = [
        metricCard('Total Users', ds.total_users, `${cs.cold_start_users} cold-start`),
        metricCard('Total Products', ds.total_products, `${cs.cold_start_products} cold-start`),
        metricCard('Interactions', ds.total_interactions.toLocaleString(), `${ds.avg_interactions_per_user} avg/user`),
        metricCard('SVD Variance', `${((cf.explained_variance || 0) * 100).toFixed(1)}%`, `${cf.n_components || '—'} latent factors`),
        metricCard('Matrix Sparsity', `${((cf.sparsity || 0) * 100).toFixed(1)}%`, `${cf.matrix_shape ? cf.matrix_shape[0] + '×' + cf.matrix_shape[1] : '—'}`),
        metricCard('Active Trends', metrics.trending?.active_trends || 0, `${metrics.trending?.hot_trends || 0} hot 🔥`),
    ].join('');

    trendingEl.innerHTML = trending.trending.slice(0, 8).map((t, i) => `
        <div class="trend-item">
            <span class="trend-item-rank">${i + 1}</span>
            <span class="trend-item-name">${t.product.name}</span>
            <span class="trend-item-score ${t.trend_signal.trend_score > 70 ? 'trend-hot' : t.trend_signal.trend_score > 30 ? 'trend-warm' : ''}" 
                  style="color: ${scoreColor(t.trend_signal.trend_score)}">
                ${t.trend_signal.trend_score.toFixed(0)}
            </span>
        </div>
    `).join('');

    coldEl.innerHTML = `
        <div class="system-section">
            <h4>Cold-Start Users</h4>
            <div class="system-row"><span class="label">Count</span><span class="value">${coldStart.cold_start_users.count}</span></div>
            ${coldStart.cold_start_users.users.slice(0, 5).map(u => `
                <div class="system-row"><span class="label">${u.user_id}</span><span class="value tag-category" style="padding:2px 8px;border-radius:12px;font-size:0.75rem">${u.strategy}</span></div>
            `).join('')}
        </div>
        <div class="system-section" style="margin-top:1rem">
            <h4>Exploration Candidates</h4>
            ${coldStart.exploration_candidates.map(p => `
                <div class="system-row"><span class="label">${p.name}</span><span class="value" style="font-size:0.8rem">${p.category}</span></div>
            `).join('')}
        </div>
    `;
}

function metricCard(label, value, sub) {
    return `<div class="metric-card"><div class="metric-label">${label}</div><div class="metric-value">${value}</div><div class="metric-sub">${sub}</div></div>`;
}

function scoreColor(score) {
    if (score > 70) return '#fca5a5';
    if (score > 30) return '#fcd34d';
    return '#67e8f9';
}

// ─── Users Dropdown ───
async function loadUsers() {
    const sel = document.getElementById('user-select');
    const data = await api('/users?limit=500');
    const coldStartData = await api('/cold-start/status');
    const coldUserIds = new Set(coldStartData.cold_start_users.users.map(u => u.user_id));

    sel.innerHTML = data.users.map(u => {
        const isCold = coldUserIds.has(u.user_id);
        return `<option value="${u.user_id}">${u.user_id} — ${u.username}${isCold ? ' ❄️' : ''}</option>`;
    }).join('');
}

// ─── Recommendations ───
document.getElementById('get-recs-btn').addEventListener('click', loadRecommendations);

async function loadRecommendations() {
    const grid = document.getElementById('recs-grid');
    const badge = document.getElementById('strategy-badge');
    const userBadge = document.getElementById('user-info-badge');
    loading(grid);

    const userId = document.getElementById('user-select').value;
    const n = parseInt(document.getElementById('num-recs').value);
    const tw = parseFloat(document.getElementById('trend-weight').value);
    const div = parseFloat(document.getElementById('diversity').value);

    // Show user info
    const userData = await api(`/users/${userId}`);
    const u = userData.user;
    const ci = userData.cold_start_info;
    userBadge.innerHTML = `
        <strong>${u.username}</strong> · Age ${u.age} · ${u.gender} · 
        ${userData.interaction_count} interactions
        ${u.preferred_categories.map(c => `<span class="tag">${c}</span>`).join('')}
        ${ci.is_cold_start ? `<span class="tag" style="background:rgba(6,182,212,0.15);color:#67e8f9">❄️ ${ci.strategy}</span>` : ''}
    `;

    const data = await apiPost('/recommendations', {
        user_id: userId,
        num_recommendations: n,
        include_trending: true,
        trending_weight: tw,
        diversity_factor: div,
    });

    // Strategy badge
    const strat = data.strategy_used;
    badge.className = `strategy-badge show ${strat.includes('cold') ? 'cold_start' : strat.includes('content') ? 'content' : 'hybrid'}`;
    badge.innerHTML = `
        <strong>Strategy:</strong> ${strat.replace(/_/g, ' ')} · 
        ${data.total_candidates_evaluated} candidates evaluated · 
        ${data.recommendations.length} returned
    `;

    grid.innerHTML = data.recommendations.map(rec => {
        const e = rec.explanation;
        return `
        <div class="rec-card">
            <div class="rec-rank">${rec.rank}</div>
            <div class="rec-name">${rec.product.name}</div>
            <div class="rec-meta">
                <span class="tag tag-category">${rec.product.category}</span>
                <span class="tag tag-brand">${rec.product.brand}</span>
                <span class="tag tag-price">$${rec.product.price.toFixed(2)}</span>
            </div>
            <div class="rec-scores">
                <div class="score-item"><div class="score-label">Collab</div><div class="score-val collab">${(e.collaborative_score * 100).toFixed(0)}</div></div>
                <div class="score-item"><div class="score-label">Content</div><div class="score-val content">${(e.content_score * 100).toFixed(0)}</div></div>
                <div class="score-item"><div class="score-label">Trend</div><div class="score-val trend">${(e.trend_score * 100).toFixed(0)}</div></div>
            </div>
            <div class="contrib-bar">
                <div class="contrib-personal" style="width:${e.personalization_contribution_pct}%"></div>
                <div class="contrib-trend" style="width:${e.trend_contribution_pct}%"></div>
            </div>
            <div class="rec-explanation">
                ${e.reasons.map(r => `<div class="reason">${r}</div>`).join('')}
            </div>
        </div>`;
    }).join('');
}

// ─── Trending ───
document.getElementById('get-trending-btn').addEventListener('click', loadTrending);

async function loadTrending() {
    const grid = document.getElementById('trending-grid');
    loading(grid);

    const cat = document.getElementById('trend-category').value;
    const window = document.getElementById('trend-window').value;
    const params = `?window_hours=${window}&top_n=20${cat ? `&category=${encodeURIComponent(cat)}` : ''}`;
    const data = await api(`/trending${params}`);

    grid.innerHTML = data.trending.map(t => {
        const score = t.trend_signal.trend_score;
        const cls = score > 70 ? 'trend-hot' : score > 30 ? 'trend-warm' : 'trend-mild';
        const gradient = score > 70
            ? 'linear-gradient(90deg, #ef4444, #f59e0b)'
            : score > 30
            ? 'linear-gradient(90deg, #f59e0b, #fcd34d)'
            : 'linear-gradient(90deg, #06b6d4, #67e8f9)';

        return `
        <div class="trend-card">
            <span class="trend-score-badge ${cls}">🔥 ${score.toFixed(0)}</span>
            <div style="font-weight:700;font-size:1rem;margin-bottom:0.25rem">${t.product.name}</div>
            <div class="rec-meta">
                <span class="tag tag-category">${t.product.category}</span>
                <span class="tag tag-brand">${t.product.brand}</span>
                <span class="tag tag-price">$${t.product.price.toFixed(2)}</span>
            </div>
            <div style="font-size:0.8rem;color:var(--text-secondary);margin-top:0.5rem">
                Velocity: ${t.trend_signal.velocity.toFixed(2)}/hr · 
                ${t.trend_signal.interaction_count_window} interactions in ${t.trend_signal.window_hours}h
            </div>
            <div class="trend-bar-bg"><div class="trend-bar-fill" style="width:${score}%;background:${gradient}"></div></div>
        </div>`;
    }).join('');
}

// ─── Categories ───
async function loadCategories() {
    const data = await api('/categories');
    const sel = document.getElementById('trend-category');
    sel.innerHTML = '<option value="">All Categories</option>' +
        data.categories.map(c => `<option value="${c.name}">${c.name} (${c.product_count})</option>`).join('');
}

// ─── System Tab ───
async function loadSystem() {
    const el = document.getElementById('system-details');
    loading(el);

    const [metrics, evolution] = await Promise.all([
        api('/system/metrics'),
        api('/system/evolution'),
    ]);

    const ret = evolution.retrain;
    const tl = evolution.trend_lifecycle;
    const ow = evolution.over_amplification_warnings;

    el.innerHTML = `
    <div class="system-grid">
        <div class="card">
            <h3 class="card-title">🧠 Model Training</h3>
            <div class="system-section">
                <h4>Collaborative Filtering (SVD)</h4>
                ${sysRow('Matrix Size', metrics.training_metrics?.collaborative?.matrix_shape?.join(' × '))}
                ${sysRow('Latent Factors', metrics.training_metrics?.collaborative?.n_components)}
                ${sysRow('Explained Variance', ((metrics.training_metrics?.collaborative?.explained_variance || 0) * 100).toFixed(1) + '%')}
                ${sysRow('Sparsity', ((metrics.training_metrics?.collaborative?.sparsity || 0) * 100).toFixed(1) + '%')}
            </div>
            <div class="system-section">
                <h4>Content-Based (TF-IDF)</h4>
                ${sysRow('Features', metrics.training_metrics?.content_based?.n_features)}
                ${sysRow('Vocabulary', metrics.training_metrics?.content_based?.vocabulary_size)}
            </div>
        </div>
        <div class="card">
            <h3 class="card-title">📈 Trend Lifecycle</h3>
            <div class="system-section">
                <h4>Configuration</h4>
                ${sysRow('Decay Rate', tl.decay_rate)}
                ${sysRow('Max Lifetime', tl.max_trend_lifetime_hours + 'h')}
                ${sysRow('Active Trends (>50)', tl.active_trends_above_50)}
            </div>
            <div class="system-section">
                <h4>Retrain Status</h4>
                ${sysRow('Should Retrain', ret.should_retrain ? '<span class="status-warn">Yes</span>' : '<span class="status-good">No</span>')}
                ${sysRow('Reason', ret.reason)}
                ${sysRow('Last Retrain', ret.last_retrain ? new Date(ret.last_retrain).toLocaleString() : '—')}
            </div>
        </div>
        <div class="card">
            <h3 class="card-title">⚠️ Over-Amplification Warnings</h3>
            ${ow.length === 0
                ? '<div style="color:var(--accent-5);font-size:0.9rem;padding:1rem 0">✅ No over-amplification detected</div>'
                : ow.map(w => `
                    <div style="padding:0.75rem;background:rgba(239,68,68,0.1);border-radius:8px;margin-bottom:0.5rem;font-size:0.85rem">
                        <strong>${w.product_id}</strong> — Score: ${w.trend_score} → Suggested: ${w.suggested_score}<br>
                        <span style="color:var(--text-secondary)">${w.recommendation}</span>
                    </div>
                `).join('')}
        </div>
        <div class="card">
            <h3 class="card-title">📊 Interaction Distribution</h3>
            ${Object.entries(metrics.interaction_distribution || {}).map(([type, count]) => `
                <div class="system-row">
                    <span class="label">${type}</span>
                    <span class="value">${count.toLocaleString()}</span>
                </div>
            `).join('')}
        </div>
    </div>`;
}

function sysRow(label, value) {
    return `<div class="system-row"><span class="label">${label}</span><span class="value">${value || '—'}</span></div>`;
}

// ─── Nav tab handlers for lazy loading ───
document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        if (btn.dataset.tab === 'system') loadSystem();
        if (btn.dataset.tab === 'trending') loadTrending();
    });
});

// ─── Init ───
(async () => {
    await Promise.all([loadDashboard(), loadUsers(), loadCategories()]);
})();
