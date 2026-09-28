/**
 * ORION — Personal Intelligence Workspace
 * Interactive Architecture & Astronomical Canvas Engine
 */

(function () {
  'use strict';

  // State Management
  const state = {
    audioEnabled: true,
    activeRailView: 'home',
    activePill: 'memory-graph',
    fluidMode: false,
    particles: [],
    constellations: [],
    orbits: []
  };

  // DOM Elements
  const scaler = document.getElementById('viewportScaler');
  const frame = document.getElementById('orionFrame');
  const canvas = document.getElementById('astronomicalCanvas');
  const ctx = canvas.getContext('2d');
  const commandInput = document.getElementById('commandInput');
  const commandBarContainer = document.getElementById('commandBarContainer');
  const submitBtn = document.getElementById('submitBtn');
  const attachBtn = document.getElementById('attachBtn');
  const voiceBtn = document.getElementById('voiceBtn');
  const liveDateTime = document.getElementById('liveDateTime');
  const modalBackdrop = document.getElementById('modalBackdrop');
  const modalTitle = document.getElementById('modalTitle');
  const modalBody = document.getElementById('modalBody');
  const modalCloseBtn = document.getElementById('modalCloseBtn');
  const modalActionBtn = document.getElementById('modalActionBtn');
  const vpFitBtn = document.getElementById('vpFitBtn');
  const vpFillBtn = document.getElementById('vpFillBtn');
  const soundToggleBtn = document.getElementById('soundToggleBtn');
  const themeToggleBtn = document.getElementById('themeToggleBtn');
  const searchTriggerBtn = document.getElementById('searchTriggerBtn');

  // =========================================================================
  // 1. Audio Engine (Web Audio API - Subtle Ethereal Chimes)
  // =========================================================================
  let audioCtx = null;

  function initAudio() {
    if (!audioCtx && (window.AudioContext || window.webkitAudioContext)) {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
  }

  function playGentleChime(type = 'submit') {
    if (!state.audioEnabled) return;
    try {
      initAudio();
      if (!audioCtx) return;
      if (audioCtx.state === 'suspended') {
        audioCtx.resume();
      }

      const now = audioCtx.currentTime;
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();

      osc.type = 'sine';
      if (type === 'submit') {
        osc.frequency.setValueAtTime(587.33, now); // D5
        osc.frequency.exponentialRampToValueAtTime(880.00, now + 0.18); // A5
        gain.gain.setValueAtTime(0.04, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.45);
      } else if (type === 'focus') {
        osc.frequency.setValueAtTime(440.00, now); // A4
        gain.gain.setValueAtTime(0.02, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.25);
      } else if (type === 'action') {
        osc.frequency.setValueAtTime(659.25, now); // E5
        osc.frequency.exponentialRampToValueAtTime(783.99, now + 0.15); // G5
        gain.gain.setValueAtTime(0.03, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.35);
      }

      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start(now);
      osc.stop(now + 0.5);
    } catch (e) {
      // Audio autoplay policy fallback
    }
  }

  // =========================================================================
  // 2. Viewport Scaler (1600x900 Aspect-Ratio Engine)
  // =========================================================================
  function updateViewportScale() {
    if (state.fluidMode) {
      frame.style.transform = 'none';
      return;
    }

    const targetWidth = 1600;
    const targetHeight = 900;
    const winWidth = window.innerWidth;
    const winHeight = window.innerHeight;

    // Calculate scale factor with slight margin padding
    const scaleX = (winWidth - 24) / targetWidth;
    const scaleY = (winHeight - 24) / targetHeight;
    const scale = Math.min(scaleX, scaleY, 1.15); // Allow slight scale up on 4k, down on smaller

    frame.style.transform = `scale(${Math.max(0.4, scale)})`;
  }

  window.addEventListener('resize', updateViewportScale);
  updateViewportScale();

  // =========================================================================
  // 3. Astronomical Canvas Engine (Sparse Particles & Faint Orbital Curves)
  //    * Completely devoid of any human/robot figure
  //    * Pure cosmic negative space environmental texture
  // =========================================================================
  function initAstronomicalAtmosphere() {
    const count = 55;
    state.particles = [];
    for (let i = 0; i < count; i++) {
      state.particles.push({
        x: Math.random() * 1600,
        y: Math.random() * 900,
        radius: Math.random() * 0.9 + 0.5,
        baseAlpha: Math.random() * 0.28 + 0.12,
        twinkleSpeed: Math.random() * 0.02 + 0.008,
        twinkleOffset: Math.random() * Math.PI * 2,
        vx: (Math.random() - 0.5) * 0.12,
        vy: (Math.random() - 0.5) * 0.08
      });
    }

    // Faint constellation points in central-upper negative space
    state.constellations = [
      { x: 780, y: 220, alpha: 0.25 },
      { x: 860, y: 190, alpha: 0.35 },
      { x: 940, y: 240, alpha: 0.22 },
      { x: 910, y: 310, alpha: 0.3 },
      { x: 810, y: 330, alpha: 0.28 },
      { x: 860, y: 410, alpha: 0.2 }
    ];

    // Orbital curves: Faint celestial ellipses centered around depth
    state.orbits = [
      {
        cx: 860,
        cy: 330,
        rx: 270,
        ry: 115,
        rotation: -0.22,
        angleOffset: 0,
        speed: 0.0003,
        alpha: 0.085,
        color: '154, 124, 255'
      },
      {
        cx: 860,
        cy: 330,
        rx: 360,
        ry: 155,
        rotation: 0.15,
        angleOffset: Math.PI / 3,
        speed: -0.00022,
        alpha: 0.06,
        color: '101, 123, 255'
      },
      {
        cx: 860,
        cy: 330,
        rx: 460,
        ry: 195,
        rotation: -0.08,
        angleOffset: Math.PI / 1.5,
        speed: 0.00018,
        alpha: 0.045,
        color: '182, 154, 255'
      }
    ];
  }

  function renderAstronomicalCanvas(time) {
    ctx.clearRect(0, 0, 1600, 900);

    // 1. Draw subtle orbital arcs
    state.orbits.forEach(orbit => {
      orbit.angleOffset += orbit.speed;
      ctx.save();
      ctx.translate(orbit.cx, orbit.cy);
      ctx.rotate(orbit.rotation);

      // Faint elliptical orbit track
      ctx.beginPath();
      ctx.ellipse(0, 0, orbit.rx, orbit.ry, 0, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(${orbit.color}, ${orbit.alpha})`;
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 12]);
      ctx.stroke();
      ctx.setLineDash([]);

      // Very small subtle orbital node marker (delicate 1.5px starlight node)
      const nodeX = Math.cos(orbit.angleOffset) * orbit.rx;
      const nodeY = Math.sin(orbit.angleOffset) * orbit.ry;
      ctx.beginPath();
      ctx.arc(nodeX, nodeY, 1.4, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${orbit.color}, ${orbit.alpha * 3.5})`;
      ctx.shadowColor = `rgba(${orbit.color}, 0.5)`;
      ctx.shadowBlur = 6;
      ctx.fill();
      ctx.shadowBlur = 0;

      ctx.restore();
    });

    // 2. Draw faint constellation lines
    ctx.beginPath();
    for (let i = 0; i < state.constellations.length - 1; i++) {
      const p1 = state.constellations[i];
      const p2 = state.constellations[i + 1];
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
    }
    ctx.strokeStyle = 'rgba(154, 124, 255, 0.04)';
    ctx.lineWidth = 0.8;
    ctx.stroke();

    // Draw constellation nodes
    state.constellations.forEach(pt => {
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 1.3, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(182, 154, 255, ${pt.alpha})`;
      ctx.shadowColor = 'rgba(154, 124, 255, 0.4)';
      ctx.shadowBlur = 4;
      ctx.fill();
      ctx.shadowBlur = 0;
    });

    // 3. Draw sparse floating particles
    const sec = time * 0.001;
    state.particles.forEach(p => {
      p.x += p.vx;
      p.y += p.vy;

      if (p.x < 0) p.x = 1600;
      if (p.x > 1600) p.x = 0;
      if (p.y < 0) p.y = 900;
      if (p.y > 900) p.y = 0;

      const alpha = p.baseAlpha + Math.sin(sec * 3 + p.twinkleOffset) * 0.08;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(244, 243, 248, ${Math.max(0.04, alpha)})`;
      ctx.fill();
    });

    requestAnimationFrame(renderAstronomicalCanvas);
  }

  initAstronomicalAtmosphere();
  requestAnimationFrame(renderAstronomicalCanvas);

  // =========================================================================
  // 4. Live Date & Time Engine
  // =========================================================================
  function updateLiveDateTime() {
    const now = new Date();
    const options = { weekday: 'short', day: 'numeric', month: 'short', year: 'numeric' };
    const dateStr = now.toLocaleDateString('en-US', options);
    const hours = String(now.getHours()).padStart(2, '0');
    const minutes = String(now.getMinutes()).padStart(2, '0');
    liveDateTime.textContent = `${dateStr} • ${hours}:${minutes}`;
  }
  updateLiveDateTime();
  setInterval(updateLiveDateTime, 30000);

  // =========================================================================
  // 5. Command Center Interactive Handling
  // =========================================================================
  commandInput.addEventListener('focus', () => {
    commandBarContainer.classList.add('focused');
    playGentleChime('focus');
  });

  commandInput.addEventListener('blur', () => {
    commandBarContainer.classList.remove('focused');
  });

  function executeCommand() {
    const query = commandInput.value.trim();
    if (!query) {
      commandInput.focus();
      return;
    }

    playGentleChime('submit');

    // Show intelligent contextual response in modal
    modalTitle.textContent = `Orion Intelligence — Query Execution`;
    modalBody.innerHTML = `
      <div style="display: flex; flex-direction: column; gap: 16px;">
        <div style="display: flex; align-items: center; gap: 10px; padding: 12px 16px; background: rgba(154, 124, 255, 0.08); border-radius: 12px; border: 1px solid rgba(154, 124, 255, 0.2);">
          <span style="color: var(--accent-violet-bright); font-size: 16px;">✦</span>
          <span style="font-weight: 500; color: var(--text-primary);">"${escapeHtml(query)}"</span>
        </div>
        
        <p style="color: var(--text-secondary); line-height: 1.65;">
          Synthesizing across active memory graph, recent project context (<strong style="color: var(--text-primary)">DAA Notes</strong>, <strong style="color: var(--text-primary)">QuOra</strong>, <strong style="color: var(--text-primary)">Arch Setup</strong>), and personal knowledge graph.
        </p>

        <div style="background: rgba(0,0,0,0.3); border-radius: 12px; padding: 14px 16px; border: 1px solid rgba(255,255,255,0.06); font-family: var(--font-body); font-size: 13px;">
          <div style="color: var(--status-success); margin-bottom: 6px; display: flex; align-items: center; gap: 6px;">
            <span style="width: 6px; height: 6px; border-radius: 50%; background: var(--status-success); display: inline-block;"></span>
            <span>Intention parsed successfully</span>
          </div>
          <div style="color: var(--text-muted);">
            • Context link: Floyd Warshall recurrence &lt;d(k)[i][j] = min(d(k-1)[i][j], d(k-1)[i][k] + d(k-1)[k][j])&gt;<br>
            • Action queue: Added to today's review schedule at 21:00.<br>
            • Knowledge edge: Correlated with Dynamic Programming matrix optimization.
          </div>
        </div>
      </div>
    `;

    openModal();
    commandInput.value = '';
    commandInput.blur();
  }

  commandInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      executeCommand();
    }
  });

  submitBtn.addEventListener('click', executeCommand);

  attachBtn.addEventListener('click', () => {
    playGentleChime('action');
    modalTitle.textContent = `Attach Context to Orion`;
    modalBody.innerHTML = `
      <p style="margin-bottom: 16px;">Select context source to attach into Orion's active attention window:</p>
      <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">
        <button class="action-tile" style="align-items: flex-start; text-align: left; padding: 12px;" onclick="selectContext('Floyd Warshall Notes')">
          <strong style="color: var(--text-primary); font-size: 12.5px;">DAA Notes</strong>
          <span style="color: var(--text-muted); font-size: 11px;">Algorithm formulation</span>
        </button>
        <button class="action-tile" style="align-items: flex-start; text-align: left; padding: 12px;" onclick="selectContext('QuOra Architecture')">
          <strong style="color: var(--text-primary); font-size: 12.5px;">QuOra Project</strong>
          <span style="color: var(--text-muted); font-size: 11px;">Quantum oracle ideas</span>
        </button>
        <button class="action-tile" style="align-items: flex-start; text-align: left; padding: 12px;" onclick="selectContext('Arch Setup Guide')">
          <strong style="color: var(--text-primary); font-size: 12.5px;">Arch Linux Guide</strong>
          <span style="color: var(--text-muted); font-size: 11px;">Hyprland & config</span>
        </button>
        <button class="action-tile" style="align-items: flex-start; text-align: left; padding: 12px;" onclick="selectContext('Local Workspace Files')">
          <strong style="color: var(--text-primary); font-size: 12.5px;">Local Workspace</strong>
          <span style="color: var(--text-muted); font-size: 11px;">Browse local files</span>
        </button>
      </div>
    `;
    openModal();
  });

  window.selectContext = function(name) {
    closeModal();
    commandInput.value = `Analyze context: [${name}] `;
    commandInput.focus();
    playGentleChime('action');
  };

  voiceBtn.addEventListener('click', () => {
    playGentleChime('action');
    voiceBtn.classList.toggle('active');
    if (voiceBtn.classList.contains('active')) {
      voiceBtn.style.color = 'var(--status-error)';
      commandInput.placeholder = "Listening... Speak your intention";
      commandInput.focus();
    } else {
      voiceBtn.style.color = '';
      commandInput.placeholder = "What’s on your mind?";
    }
  });

  // =========================================================================
  // 6. Pill Shortcuts Below Command Bar
  // =========================================================================
  const pills = document.querySelectorAll('.shortcut-pill');
  pills.forEach(pill => {
    pill.addEventListener('click', () => {
      pills.forEach(p => p.classList.remove('active-pill'));
      pill.classList.add('active-pill');
      state.activePill = pill.dataset.pill;
      playGentleChime('action');

      const pillNames = {
        'memory-graph': 'Memory Graph View',
        'my-tasks': 'Active Tasks & Intentions',
        'knowledge-base': 'Synthesized Knowledge Base',
        'web-search': 'Real-time Web Search Grounding',
        'tools': 'Tool & Plugin Integrations'
      };

      modalTitle.textContent = pillNames[state.activePill] || 'Workspace Navigation';
      modalBody.innerHTML = `
        <div style="padding: 10px 0;">
          <h4 style="color: var(--text-primary); margin-bottom: 8px; font-weight: 500;">
            ${pillNames[state.activePill]}
          </h4>
          <p style="color: var(--text-secondary); margin-bottom: 16px;">
            Orion has indexed this domain with active context from your personal workspace.
          </p>
          <div style="display: flex; gap: 8px; flex-wrap: wrap;">
            <span class="kbd-badge" style="padding: 4px 10px; font-size: 12px; color: var(--accent-violet-bright);">
              ● Ready for query
            </span>
            <span class="kbd-badge" style="padding: 4px 10px; font-size: 12px;">
              Latency: 18ms
            </span>
            <span class="kbd-badge" style="padding: 4px 10px; font-size: 12px;">
              Encrypted Local Session
            </span>
          </div>
        </div>
      `;
      openModal();
    });
  });

  // =========================================================================
  // 7. Left Sidebar Navigation
  // =========================================================================
  const railItems = document.querySelectorAll('.sidebar-nav-group .rail-item');
  railItems.forEach(item => {
    item.addEventListener('click', () => {
      railItems.forEach(r => r.classList.remove('active'));
      item.classList.add('active');
      state.activeRailView = item.dataset.view;
      playGentleChime('action');
    });
  });

  // =========================================================================
  // 8. Quick Actions Interactivity
  // =========================================================================
  const actionTiles = document.querySelectorAll('.action-tile');
  actionTiles.forEach(tile => {
    tile.addEventListener('click', () => {
      const action = tile.dataset.action;
      playGentleChime('action');

      if (action === 'new-note') {
        modalTitle.textContent = "New Note — Orion Workspace";
        modalBody.innerHTML = `
          <div style="display: flex; flex-direction: column; gap: 14px;">
            <input type="text" placeholder="Note title..." style="font-size: 18px; font-weight: 500; color: var(--text-primary); border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 8px;">
            <textarea placeholder="Write or capture thoughts... Orion will automatically extract entities, tags, and cross-references." style="width: 100%; height: 160px; background: transparent; border: none; outline: none; color: var(--text-secondary); font-family: var(--font-body); font-size: 14px; resize: none;"></textarea>
          </div>
        `;
      } else if (action === 'plan-day') {
        modalTitle.textContent = "Plan My Day — Intelligent Agenda";
        modalBody.innerHTML = `
          <div style="display: flex; flex-direction: column; gap: 12px;">
            <p style="color: var(--text-secondary);">Orion prioritized your key tasks based on deadlines and cognitive energy:</p>
            <div style="display: flex; align-items: center; gap: 10px; padding: 10px 14px; background: rgba(255,255,255,0.03); border-radius: 10px; border: 1px solid rgba(255,255,255,0.06);">
              <span style="color: var(--accent-violet-bright);">1.</span>
              <span style="color: var(--text-primary); font-weight: 480;">Review DAA Floyd Warshall edge cases</span>
              <span style="margin-left: auto; color: var(--status-warning); font-size: 12px;">High Priority</span>
            </div>
            <div style="display: flex; align-items: center; gap: 10px; padding: 10px 14px; background: rgba(255,255,255,0.03); border-radius: 10px; border: 1px solid rgba(255,255,255,0.06);">
              <span style="color: var(--accent-violet-bright);">2.</span>
              <span style="color: var(--text-primary); font-weight: 480;">Draft QuOra architecture proposal</span>
              <span style="margin-left: auto; color: var(--text-muted); font-size: 12px;">Afternoon</span>
            </div>
            <div style="display: flex; align-items: center; gap: 10px; padding: 10px 14px; background: rgba(255,255,255,0.03); border-radius: 10px; border: 1px solid rgba(255,255,255,0.06);">
              <span style="color: var(--accent-violet-bright);">3.</span>
              <span style="color: var(--text-primary); font-weight: 480;">Test Arch Linux pipewire configuration</span>
              <span style="margin-left: auto; color: var(--text-muted); font-size: 12px;">Evening</span>
            </div>
          </div>
        `;
      } else if (action === 'explain-this') {
        modalTitle.textContent = "Explain This — Concept Deconstruction";
        modalBody.innerHTML = `
          <p style="margin-bottom: 12px;">Enter any concept, equation, or code fragment to deconstruct intuitively:</p>
          <input type="text" value="Floyd-Warshall algorithm 3D state recurrence" style="width: 100%; padding: 10px 14px; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; color: var(--text-primary); margin-bottom: 12px;">
          <div style="padding: 12px; background: rgba(154, 124, 255, 0.08); border-radius: 10px; color: var(--text-secondary); font-size: 13px; line-height: 1.6;">
            The Floyd-Warshall algorithm is an all-pairs shortest path dynamic programming algorithm that operates in O(V³) time. At each step k, it checks whether passing through vertex k offers a shorter path between any pair (i, j).
          </div>
        `;
      } else if (action === 'brainstorm') {
        modalTitle.textContent = "Brainstorming Canvas — Orion Ideation";
        modalBody.innerHTML = `
          <p style="margin-bottom: 12px;">What would you like to explore or synthesize together?</p>
          <div style="display: flex; flex-direction: column; gap: 8px;">
            <button class="action-tile" style="align-items: flex-start; text-align: left; padding: 10px;" onclick="selectContext('Quantum ML Hybrid Architectures')">
              <span style="color: var(--text-primary); font-weight: 500;">✦ Quantum ML Hybrid Architectures</span>
              <span style="color: var(--text-muted); font-size: 11px;">Parameter-shift gradients and classical feature fusion</span>
            </button>
            <button class="action-tile" style="align-items: flex-start; text-align: left; padding: 10px;" onclick="selectContext('Product Design for Personal Intelligence')">
              <span style="color: var(--text-primary); font-weight: 500;">✦ Ambient Workspace Experience</span>
              <span style="color: var(--text-muted); font-size: 11px;">Calm interfaces without conversational friction</span>
            </button>
          </div>
        `;
      }

      openModal();
    });
  });

  // =========================================================================
  // 9. Recent Items Interactivity
  // =========================================================================
  const recentItems = document.querySelectorAll('.recent-item');
  recentItems.forEach(item => {
    item.addEventListener('click', () => {
      const type = item.dataset.recent;
      playGentleChime('action');

      if (type === 'floyd-warshall') {
        modalTitle.textContent = "DAA — Floyd Warshall Notes";
        modalBody.innerHTML = `
          <div style="display: flex; flex-direction: column; gap: 12px;">
            <div style="display: flex; justify-content: space-between; font-size: 12px; color: var(--text-muted);">
              <span>Updated 2 hours ago</span>
              <span>Tags: #algorithms #graphs #dynamic-programming</span>
            </div>
            <p><strong>Recurrence Relation:</strong></p>
            <pre style="background: rgba(0,0,0,0.35); padding: 12px; border-radius: 8px; font-family: monospace; font-size: 12.5px; color: var(--accent-violet-bright); overflow-x: auto;">dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);</pre>
            <p style="line-height: 1.6; color: var(--text-secondary);">
              Iterating k from 1 to V. Detects negative weight cycles if dist[i][i] &lt; 0 for any vertex i. Memory footprint optimized from O(V³) to O(V²) space by reusing the distance matrix.
            </p>
          </div>
        `;
      } else if (type === 'quora') {
        modalTitle.textContent = "QuOra — Project Ideas";
        modalBody.innerHTML = `
          <div style="display: flex; flex-direction: column; gap: 12px;">
            <div style="display: flex; justify-content: space-between; font-size: 12px; color: var(--text-muted);">
              <span>Updated 5 hours ago</span>
              <span>Tags: #quantum #architecture #hybrid-kernel</span>
            </div>
            <p style="line-height: 1.6; color: var(--text-secondary);">
              High-dimensional feature space mappings using Angle Embedding with PennyLane and Qiskit. Testing classical SVM vs. Quantum Kernel estimation on medical biomarker classification.
            </p>
          </div>
        `;
      } else if (type === 'arch-linux') {
        modalTitle.textContent = "Arch Linux Setup Guide";
        modalBody.innerHTML = `
          <div style="display: flex; flex-direction: column; gap: 12px;">
            <div style="display: flex; justify-content: space-between; font-size: 12px; color: var(--text-muted);">
              <span>Updated 1 day ago</span>
              <span>Tags: #linux #hyprland #wayland #dotfiles</span>
            </div>
            <p style="line-height: 1.6; color: var(--text-secondary);">
              Clean Hyprland configuration with blur decorations, Catppuccin mocha violet accents, waybar status metrics, and Pipewire audio low-latency routing.
            </p>
          </div>
        `;
      }

      openModal();
    });
  });

  document.getElementById('seeAllActionsBtn').addEventListener('click', () => {
    actionTiles[0].click();
  });

  document.getElementById('seeAllRecentBtn').addEventListener('click', () => {
    recentItems[0].click();
  });

  // =========================================================================
  // 10. Modal Dialog Open / Close Helpers
  // =========================================================================
  function openModal() {
    modalBackdrop.classList.add('open');
  }

  function closeModal() {
    modalBackdrop.classList.remove('open');
  }

  modalCloseBtn.addEventListener('click', closeModal);
  modalActionBtn.addEventListener('click', closeModal);

  modalBackdrop.addEventListener('click', (e) => {
    if (e.target === modalBackdrop) {
      closeModal();
    }
  });

  // Keyboard navigation & shortcuts
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      closeModal();
    }
    // Search hotkey: Cmd+K or Ctrl+K or /
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
      e.preventDefault();
      commandInput.focus();
    }
    if (e.key === '/' && document.activeElement !== commandInput && document.activeElement.tagName !== 'INPUT' && document.activeElement.tagName !== 'TEXTAREA') {
      e.preventDefault();
      commandInput.focus();
    }
  });

  searchTriggerBtn.addEventListener('click', () => {
    commandInput.focus();
    playGentleChime('focus');
  });

  // =========================================================================
  // 11. Viewport & Setting Controls
  // =========================================================================
  vpFitBtn.addEventListener('click', () => {
    state.fluidMode = false;
    document.body.classList.remove('fluid-mode');
    vpFitBtn.classList.add('active');
    vpFillBtn.classList.remove('active');
    updateViewportScale();
    playGentleChime('action');
  });

  vpFillBtn.addEventListener('click', () => {
    state.fluidMode = true;
    document.body.classList.add('fluid-mode');
    vpFillBtn.classList.add('active');
    vpFitBtn.classList.remove('active');
    updateViewportScale();
    playGentleChime('action');
  });

  soundToggleBtn.addEventListener('click', () => {
    state.audioEnabled = !state.audioEnabled;
    soundToggleBtn.classList.toggle('active', state.audioEnabled);
    soundToggleBtn.title = `Synthesizer audio feedback: ${state.audioEnabled ? 'On' : 'Off'}`;
    if (state.audioEnabled) {
      playGentleChime('action');
    }
  });

  themeToggleBtn.addEventListener('click', () => {
    playGentleChime('action');
    modalTitle.textContent = "Appearance: Cosmic Dark";
    modalBody.innerHTML = `
      <p style="color: var(--text-secondary);">
        ORION is rendered in curated <strong>Cosmic Dark (#090A11)</strong> with violet spectral luminescence and 24px backdrop diffusion. It is optimized for zero eyestrain and deep focus.
      </p>
    `;
    openModal();
  });

  // Utility
  function escapeHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

})();
