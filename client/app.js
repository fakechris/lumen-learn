/**
 * Socratic Whiteboard Frontend Client Controller
 * Manages Multi-column Whiteboard, KaTeX formulas, SVG Hand-drawn Annotations,
 * 3D Three.js Sandbox, WebAudio Typewriter Sync, and Socratic Interactions.
 */

class WhiteboardApp {
  constructor() {
    this.currentCourseId = null;
    this.currentCourseStructure = null;
    this.currentSessionId = null;
    this.currentSessionManifest = null;
    this.currentStepIndex = 0;
    this.isPlaying = false;
    this.playbackSpeed = 1.25;
    this.audioElement = new Audio();
    this.rafId = null;
    this.activeTab = "dialogue";

    this.initElements();
    this.bindEvents();
    this.loadInitialCourse();
  }

  initElements() {
    this.courseTitlePill = document.getElementById("courseTitlePill");
    this.sessionCounter = document.getElementById("sessionCounter");
    this.prevSessionBtn = document.getElementById("prevSessionBtn");
    this.nextSessionBtn = document.getElementById("nextSessionBtn");
    this.speedToggleBtn = document.getElementById("speedToggleBtn");
    
    this.columnLeft = document.getElementById("columnLeft");
    this.columnRight = document.getElementById("columnRight");
    this.whiteboardViewport = document.getElementById("whiteboardViewport");

    this.playPauseBtn = document.getElementById("playPauseBtn");
    this.playIcon = document.getElementById("playIcon");
    this.subtitleText = document.getElementById("subtitleText");
    this.socraticOptionsRow = document.getElementById("socraticOptionsRow");
    this.interjectBtn = document.getElementById("interjectBtn");

    this.tabDialogue = document.getElementById("tabDialogue");
    this.tabOutline = document.getElementById("tabOutline");
    this.tabInspector = document.getElementById("tabInspector");
    this.sidebarContent = document.getElementById("sidebarContent");

    this.uploadModal = document.getElementById("uploadModal");
    this.settingsModal = document.getElementById("settingsModal");
    this.openUploadModalBtn = document.getElementById("openUploadModalBtn");
    this.openSettingsModalBtn = document.getElementById("openSettingsModalBtn");
    this.closeUploadModalBtn = document.getElementById("closeUploadModalBtn");
    this.closeSettingsModalBtn = document.getElementById("closeSettingsModalBtn");
    this.cancelUploadBtn = document.getElementById("cancelUploadBtn");
    this.closeSettingsBtn = document.getElementById("closeSettingsBtn");
    this.startGenerateBtn = document.getElementById("startGenerateBtn");
  }

  bindEvents() {
    this.playPauseBtn.addEventListener("click", () => this.togglePlayback());
    this.speedToggleBtn.addEventListener("click", () => this.cycleSpeed());
    this.prevSessionBtn.addEventListener("click", () => this.navigateSession(-1));
    this.nextSessionBtn.addEventListener("click", () => this.navigateSession(1));

    this.tabDialogue.addEventListener("click", () => this.switchTab("dialogue"));
    this.tabOutline.addEventListener("click", () => this.switchTab("outline"));
    this.tabInspector.addEventListener("click", () => this.switchTab("inspector"));

    this.openUploadModalBtn.addEventListener("click", () => this.uploadModal.style.display = "flex");
    this.closeUploadModalBtn.addEventListener("click", () => this.uploadModal.style.display = "none");
    this.cancelUploadBtn.addEventListener("click", () => this.uploadModal.style.display = "none");

    this.openSettingsModalBtn.addEventListener("click", () => this.settingsModal.style.display = "flex");
    this.closeSettingsModalBtn.addEventListener("click", () => this.settingsModal.style.display = "none");
    this.closeSettingsBtn.addEventListener("click", () => {
      this.saveSettings();
      this.settingsModal.style.display = "none";
    });

    this.startGenerateBtn.addEventListener("click", () => this.handleGenerateCourse());
    this.interjectBtn.addEventListener("click", () => this.handleInterjection());

    this.audioElement.addEventListener("ended", () => {
      this.isPlaying = false;
      this.updatePlayPauseIcon();
    });
  }

  async loadInitialCourse() {
    try {
      const res = await fetch("/api/v1/courses");
      const data = await res.json();
      if (data.courses && data.courses.length > 0) {
        const firstCourse = data.courses[0];
        await this.loadCourse(firstCourse.course_id);
      }
    } catch (e) {
      console.error("Failed to load initial course:", e);
    }
  }

  async loadCourse(courseId) {
    this.currentCourseId = courseId;
    const res = await fetch(`/api/v1/courses/${courseId}`);
    this.currentCourseStructure = await res.json();

    this.courseTitlePill.textContent = `📚 ${this.currentCourseStructure.title}`;
    
    // Find all sessions in order
    this.allSessions = [];
    for (const ch of this.currentCourseStructure.chapters) {
      for (const s of ch.sessions) {
        this.allSessions.push(s);
      }
    }

    if (this.allSessions.length > 0) {
      await this.loadSession(this.allSessions[0].session_id);
    }
  }

  async loadSession(sessionId) {
    this.currentSessionId = sessionId;
    const sIndex = this.allSessions.findIndex(s => s.session_id === sessionId);
    this.sessionCounter.textContent = `${sIndex + 1} / ${this.allSessions.length}`;

    const res = await fetch(`/api/v1/courses/${this.currentCourseId}/sessions/${sessionId}`);
    this.currentSessionManifest = await res.json();
    this.currentStepIndex = 0;

    this.renderWhiteboard();
    this.renderSidebar();
    this.startStep(0);
  }

  renderWhiteboard() {
    this.columnLeft.innerHTML = "";
    this.columnRight.innerHTML = "";

    const manifest = this.currentSessionManifest;
    if (!manifest || !manifest.steps) return;

    manifest.steps.forEach((step, sIdx) => {
      // 1. Board Cards
      (step.board_cards || []).forEach(card => {
        const cardElem = document.createElement("div");
        cardElem.className = "wb-card";
        cardElem.id = `card_${card.card_id}`;

        let html = `<div class="wb-card-title"><i data-lucide="edit-3" style="width:16px;height:16px;color:#2563eb;"></i> ${card.title}</div>`;
        html += `<div class="wb-card-content">${marked.parse(card.markdown)}</div>`;

        // SVG Decorations (Red Circles / Highlights)
        (card.decorations || []).forEach(dec => {
          if (dec.kind === "circle") {
            html += `
              <svg class="annotation-circle" style="inset: 0; width: 100%; height: 100%;">
                <ellipse cx="60%" cy="50%" rx="35%" ry="35%" fill="none" stroke="${dec.color || '#e05656'}" stroke-width="2.5" stroke-linecap="round" />
              </svg>
            `;
          }
        });

        cardElem.innerHTML = html;

        if (card.column_index === 1) {
          this.columnRight.appendChild(cardElem);
        } else {
          this.columnLeft.appendChild(cardElem);
        }
      });

      // 2. 3D Interactive Manipulative Widget
      if (step.widget && step.widget.widget_type === "threejs_3d") {
        const widgetContainer = document.createElement("div");
        widgetContainer.className = "widget-container";
        widgetContainer.innerHTML = `
          <div class="widget-header">
            <span>🎲 ${step.widget.title}</span>
            <span style="font-size:0.75rem;color:#2563eb;font-weight:500;">Three.js 交互模式</span>
          </div>
          <div class="widget-iframe-wrapper">
            <iframe sandbox="allow-scripts allow-same-origin" srcdoc="${this.escapeHtml(step.widget.html_content)}"></iframe>
          </div>
        `;
        this.columnRight.appendChild(widgetContainer);
      }
    });

    // Render LaTeX Math Formulas with KaTeX
    if (window.renderMathInElement) {
      renderMathInElement(this.whiteboardViewport, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "$", right: "$", display: false },
          { left: "\\[", right: "\\]", display: true },
          { left: "\\(", right: "\\)", display: false }
        ]
      });
    }

    lucide.createIcons();
  }

  escapeHtml(str) {
    return str.replace(/&/g, "&amp;").replace(/"/g, "&quot;");
  }

  startStep(stepIdx) {
    if (!this.currentSessionManifest || stepIdx >= this.currentSessionManifest.steps.length) return;
    this.currentStepIndex = stepIdx;
    const step = this.currentSessionManifest.steps[stepIdx];

    // Subtitle typewriter animation
    this.runTypewriter(step.speech_text);

    // Audio Playback
    if (step.audio_meta && step.audio_meta.audio_url) {
      this.audioElement.src = step.audio_meta.audio_url;
      this.audioElement.playbackRate = this.playbackSpeed;
      this.audioElement.play().catch(() => {});
      this.isPlaying = true;
      this.updatePlayPauseIcon();
    }

    // Render Socratic Question Chips
    this.renderSocraticChips(step.question);
    this.renderSidebar();
  }

  runTypewriter(fullText) {
    let charIdx = 0;
    this.subtitleText.textContent = "";

    if (this.typewriterInterval) clearInterval(this.typewriterInterval);

    const speedMs = Math.max(15, Math.floor(60 / this.playbackSpeed));
    this.typewriterInterval = setInterval(() => {
      if (charIdx <= fullText.length) {
        this.subtitleText.textContent = fullText.slice(0, charIdx);
        charIdx++;
      } else {
        clearInterval(this.typewriterInterval);
      }
    }, speedMs);
  }

  renderSocraticChips(question) {
    this.socraticOptionsRow.innerHTML = "";
    if (!question || !question.options) return;

    question.options.forEach(opt => {
      const chip = document.createElement("button");
      chip.className = "socratic-chip";
      chip.innerHTML = `<span>💡</span> ${opt.text}`;

      chip.addEventListener("click", () => {
        if (opt.is_correct) {
          chip.classList.add("correct");
          chip.innerHTML = `<span>✓ 正确</span> ${opt.text}`;
          this.addDialogueMessage("Chris (学生)", opt.text, "student");
          this.addDialogueMessage("小悠 (导师)", "太棒了！你的空间直觉非常敏锐。正如你所看到的，如果第三个向量没有独立的维度分量，就只能躺在同一张纸上。", "tutor");
          setTimeout(() => {
            if (this.currentStepIndex + 1 < this.currentSessionManifest.steps.length) {
              this.startStep(this.currentStepIndex + 1);
            }
          }, 1500);
        } else {
          chip.classList.add("incorrect");
          chip.innerHTML = `<span>✗ 启发思考</span> ${opt.text}`;
          this.addDialogueMessage("Chris (学生)", opt.text, "student");
          const feedback = opt.misconception_analysis || "再仔细观察一下右侧的 3D 旋转模型：第三个向量只是躺在前面的绿色平面里，并没有离开墙面哦！";
          this.addDialogueMessage("小悠 (导师)", feedback, "tutor");
        }
      });

      this.socraticOptionsRow.appendChild(chip);
    });

    const explainChip = document.createElement("button");
    explainChip.className = "socratic-chip";
    explainChip.style.borderColor = "#94a3b8";
    explainChip.innerHTML = `<span>🔍</span> 请直接解释一下`;
    explainChip.addEventListener("click", () => {
      this.addDialogueMessage("Chris (学生)", "请直接解释一下为什么", "student");
      this.addDialogueMessage("小悠 (导师)", "好的！因为空间维度的本质是「线性无关的方向个数」，单凭数量增加但共面的向量是无法张成更高维度的。", "tutor");
    });
    this.socraticOptionsRow.appendChild(explainChip);
  }

  addDialogueMessage(sender, text, role) {
    this.dialogueHistory = this.dialogueHistory || [];
    this.dialogueHistory.push({ sender, text, role });
    if (this.activeTab === "dialogue") {
      this.renderSidebar();
    }
  }

  switchTab(tab) {
    this.activeTab = tab;
    this.tabDialogue.classList.toggle("active", tab === "dialogue");
    this.tabOutline.classList.toggle("active", tab === "outline");
    this.tabInspector.classList.toggle("active", tab === "inspector");
    this.renderSidebar();
  }

  renderSidebar() {
    this.sidebarContent.innerHTML = "";

    if (this.activeTab === "dialogue") {
      const msgs = this.dialogueHistory || [
        { sender: "小悠 (导师)", text: "同学们好！今天我们来探究《线性代数：基与维数》的核心奥秘。", role: "tutor" },
        { sender: "小悠 (导师)", text: "其实这两个向量只能在它们组成的平面的墙上活动，就像一张纸漂在空中...", role: "tutor" }
      ];
      msgs.forEach(m => {
        const bubble = document.createElement("div");
        bubble.className = `chat-bubble ${m.role}`;
        bubble.innerHTML = `
          <span class="chat-bubble-sender">${m.sender}</span>
          <div class="chat-bubble-body">${m.text}</div>
        `;
        this.sidebarContent.appendChild(bubble);
      });
      this.sidebarContent.scrollTop = this.sidebarContent.scrollHeight;
    } 
    else if (this.activeTab === "outline") {
      (this.allSessions || []).forEach((s, idx) => {
        const item = document.createElement("div");
        item.className = `outline-item ${s.session_id === this.currentSessionId ? "active" : ""}`;
        item.innerHTML = `
          <i data-lucide="${s.session_id === this.currentSessionId ? "play-circle" : "check-circle-2"}" style="width:16px;height:16px;"></i>
          <div>
            <div style="font-weight:600;">${idx + 1}. ${s.title}</div>
            <div style="font-size:0.75rem;color:#64748b;">${s.estimated_duration_min} 分钟 • ${s.learning_goal}</div>
          </div>
        `;
        item.addEventListener("click", () => this.loadSession(s.session_id));
        this.sidebarContent.appendChild(item);
      });
      lucide.createIcons();
    }
    else if (this.activeTab === "inspector") {
      const step = this.currentSessionManifest?.steps?.[this.currentStepIndex];
      this.sidebarContent.innerHTML = `
        <div style="font-size:0.8rem;font-weight:700;color:#334155;margin-bottom:6px;">📊 实时生产与步骤元数据 (JSON)</div>
        <pre style="background:#0f172a;color:#38bdf8;padding:12px;border-radius:8px;font-size:0.75rem;overflow-x:auto;max-height:400px;">${JSON.stringify(step || this.currentSessionManifest, null, 2)}</pre>
      `;
    }
  }

  togglePlayback() {
    this.isPlaying = !this.isPlaying;
    if (this.isPlaying) {
      this.audioElement.play().catch(() => {});
    } else {
      this.audioElement.pause();
    }
    this.updatePlayPauseIcon();
  }

  updatePlayPauseIcon() {
    this.playIcon.setAttribute("data-lucide", this.isPlaying ? "pause" : "play");
    lucide.createIcons();
  }

  cycleSpeed() {
    const speeds = [1.0, 1.25, 1.5, 2.0];
    const nextIdx = (speeds.indexOf(this.playbackSpeed) + 1) % speeds.length;
    this.playbackSpeed = speeds[nextIdx];
    this.audioElement.playbackRate = this.playbackSpeed;
    this.speedToggleBtn.textContent = `${this.playbackSpeed}x 速度`;
  }

  navigateSession(delta) {
    const sIndex = this.allSessions.findIndex(s => s.session_id === this.currentSessionId);
    const targetIdx = sIndex + delta;
    if (targetIdx >= 0 && targetIdx < this.allSessions.length) {
      this.loadSession(this.allSessions[targetIdx].session_id);
    }
  }

  async handleGenerateCourse() {
    const title = document.getElementById("lectureTitleInput").value;
    const content = document.getElementById("lectureContentInput").value;
    const apiKey = localStorage.getItem("hk_api_key") || "";
    const provider = localStorage.getItem("hk_provider") || "deepseek";
    const baseUrl = localStorage.getItem("hk_base_url") || "";

    if (!content.trim()) {
      alert("请输入讲义内容");
      return;
    }

    this.startGenerateBtn.textContent = "⚙️ 正在拆解大纲与生成板书...";
    this.startGenerateBtn.disabled = true;

    try {
      const res = await fetch("/api/v1/generate_course", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          title,
          content,
          api_key: apiKey || null,
          llm_provider: provider,
          base_url: baseUrl || null
        })
      });
      const data = await res.json();
      if (data.success) {
        this.uploadModal.style.display = "none";
        await this.loadCourse(data.course_id);
        alert(`🎉 课程生成成功！已解构 ${data.total_chapters} 个章节，共 ${data.total_sessions} 个互动会话。`);
      }
    } catch (e) {
      alert("生成失败: " + e.message);
    } finally {
      this.startGenerateBtn.textContent = "🚀 开始 AI 拆解与生产";
      this.startGenerateBtn.disabled = false;
    }
  }

  async handleInterjection() {
    const query = prompt("请输入你想向导师提问的数学/直观疑问：", "为什么两个向量张不成三维空间？");
    if (!query) return;

    this.addDialogueMessage("Chris (打断提问)", query, "student");
    const step = this.currentSessionManifest?.steps?.[this.currentStepIndex];
    
    try {
      const res = await fetch("/api/v1/socratic_interact", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_query: query,
          current_step_context: step?.speech_text || "",
          api_key: localStorage.getItem("hk_api_key") || null
        })
      });
      const data = await res.json();
      this.addDialogueMessage("小悠 (导师解答)", data.reply, "tutor");
    } catch (e) {
      console.error(e);
    }
  }

  saveSettings() {
    const provider = document.getElementById("llmProviderSelect").value;
    const apiKey = document.getElementById("apiKeyInput").value;
    const baseUrl = document.getElementById("baseUrlInput").value;

    localStorage.setItem("hk_provider", provider);
    if (apiKey) localStorage.setItem("hk_api_key", apiKey);
    if (baseUrl) localStorage.setItem("hk_base_url", baseUrl);

    alert("模型配置已保存！支持 DeepSeek, Claude, GPT-4o, Gemini 实时调用。");
  }
}

document.addEventListener("DOMContentLoaded", () => {
  window.app = new WhiteboardApp();
});
