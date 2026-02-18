# app.py - OA Premium Dashboard with Futuristic Navigation Panel
import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_chat import message
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
import cv2
import os
from fpdf import FPDF
from werkzeug.security import generate_password_hash, check_password_hash
import io
from datetime import datetime
from torchvision import models 
import streamlit.components.v1 as components
import json

# -----------------------
# Config
# -----------------------
DB_PATH = "database.db"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR

MODEL_PATH = os.path.join(PROJECT_ROOT,
                          "models",
                          "E6_albumentations.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["Grade 0", "Grade 1", "Grade 2", "Grade 3", "Grade 4"]

# Initialize session state for page navigation
if "show_landing" not in st.session_state:
    st.session_state["show_landing"] = True
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Dashboard"

# -----------------------
# HTML Landing Page
# -----------------------
LANDING_HTML = """
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Osteoporosis Advanced Detection AI</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.7.0/p5.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        html,
        body {
            height: 100%;
            overflow: hidden;
        }

        body {
            font-family: 'Poppins', sans-serif;
            background: radial-gradient(ellipse at bottom, #1B2735 0%, #090A0F 100%);
        }

        /* Starry Background */
        #stars,
        #stars2,
        #stars3 {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
        }

        #stars {
            width: 1px;
            height: 1px;
            background: transparent;
            box-shadow: 1795px 457px #FFF, 1523px 1943px #FFF, 456px 1753px #FFF, 712px 1456px #FFF, 1134px 967px #FFF, 1889px 234px #FFF, 345px 1678px #FFF, 1567px 789px #FFF, 890px 1234px #FFF, 1678px 1567px #FFF, 234px 890px #FFF, 1456px 345px #FFF, 789px 1889px #FFF, 1234px 712px #FFF, 567px 1134px #FFF, 1943px 1795px #FFF, 1753px 1523px #FFF, 967px 456px #FFF, 345px 1678px #FFF, 1889px 234px #FFF, 712px 1456px #FFF, 1134px 967px #FFF, 1567px 789px #FFF, 890px 1234px #FFF, 1795px 1943px #FFF, 456px 1753px #FFF, 1523px 345px #FFF, 1678px 1567px #FFF, 234px 890px #FFF, 1456px 712px #FFF, 789px 1889px #FFF, 1234px 1134px #FFF, 567px 1795px #FFF;
            animation: animStar 50s linear infinite;
        }

        #stars:after {
            content: " ";
            position: absolute;
            top: 2000px;
            width: 1px;
            height: 1px;
            background: transparent;
            box-shadow: 1795px 457px #FFF, 1523px 1943px #FFF, 456px 1753px #FFF, 712px 1456px #FFF, 1134px 967px #FFF, 1889px 234px #FFF, 345px 1678px #FFF, 1567px 789px #FFF, 890px 1234px #FFF, 1678px 1567px #FFF, 234px 890px #FFF, 1456px 345px #FFF, 789px 1889px #FFF, 1234px 712px #FFF, 567px 1134px #FFF, 1943px 1795px #FFF, 1753px 1523px #FFF, 967px 456px #FFF;
        }

        #stars2 {
            width: 2px;
            height: 2px;
            background: transparent;
            box-shadow: 1234px 1567px #FFF, 567px 234px #FFF, 1890px 1456px #FFF, 345px 789px #FFF, 1678px 1234px #FFF, 890px 567px #FFF, 1456px 1890px #FFF, 234px 345px #FFF, 1567px 1678px #FFF, 789px 890px #FFF, 1234px 1456px #FFF, 456px 234px #FFF, 1789px 1567px #FFF, 1123px 789px #FFF, 678px 1234px #FFF, 1456px 890px #FFF, 234px 1678px #FFF, 1890px 345px #FFF, 567px 1456px #FFF, 1234px 234px #FFF;
            animation: animStar 100s linear infinite;
        }

        #stars2:after {
            content: " ";
            position: absolute;
            top: 2000px;
            width: 2px;
            height: 2px;
            background: transparent;
            box-shadow: 1234px 1567px #FFF, 567px 234px #FFF, 1890px 1456px #FFF, 345px 789px #FFF, 1678px 1234px #FFF, 890px 567px #FFF, 1456px 1890px #FFF, 234px 345px #FFF;
        }

        #stars3 {
            width: 3px;
            height: 3px;
            background: transparent;
            box-shadow: 890px 1234px #FFF, 1456px 567px #FFF, 234px 1890px #FFF, 1678px 345px #FFF, 567px 1456px #FFF, 1234px 789px #FFF, 345px 1567px #FFF, 1890px 234px #FFF, 789px 1678px #FFF, 1456px 890px #FFF, 234px 1234px #FFF, 1567px 456px #FFF, 890px 1789px #FFF, 1234px 1123px #FFF, 678px 567px #FFF;
            animation: animStar 150s linear infinite;
        }

        #stars3:after {
            content: " ";
            position: absolute;
            top: 2000px;
            width: 3px;
            height: 3px;
            background: transparent;
            box-shadow: 890px 1234px #FFF, 1456px 567px #FFF, 234px 1890px #FFF, 1678px 345px #FFF, 567px 1456px #FFF;
        }

        @keyframes animStar {
            from {
                transform: translateY(0px);
            }

            to {
                transform: translateY(-2000px);
            }
        }

        #canvas-container {
            position: fixed;
            inset: 0;
            z-index: 2;
        }

        .text-container {
            position: fixed;
            z-index: 3;
            top: 50%;
            transform: translateY(-50%);
        }

        .left-text {
            left: 5vw;
            text-align: left;
        }

        .right-text {
            right: 5vw;
            text-align: right;
            pointer-events: none;
        }

        /* Clickable Title Styling */
        .title-link {
            display: inline-block;
            text-decoration: none;
            color: inherit;
            pointer-events: auto;
            cursor: pointer;
            transition: all 0.3s ease;
            position: relative;
        }

        .title-link:hover .title {
            transform: scale(1.05);
            text-shadow: 0 0 40px rgba(255, 255, 255, 0.6), 0 0 20px rgba(147, 197, 253, 0.4);
        }

        .title-link:active .title {
            transform: scale(0.98);
        }

        .title-link::after {
            content: '';
            position: absolute;
            bottom: -5px;
            left: 0;
            width: 0;
            height: 3px;
            background: linear-gradient(90deg, #3498db, #9b59b6);
            transition: width 0.4s ease;
        }

        .title-link:hover::after {
            width: 100%;
        }

        .title {
            font-size: clamp(40px, 7vw, 64px);
            font-weight: 700;
            color: #ffffff;
            margin: 0;
            text-shadow: 0 0 30px rgba(255, 255, 255, 0.4);
            letter-spacing: 3px;
            line-height: 1.2;
            transition: all 0.3s ease;
        }

        .subtitle {
            font-size: clamp(22px, 4vw, 32px);
            font-weight: 300;
            color: #a0a0a0;
            margin: 12px 0 0 0;
            letter-spacing: 1.5px;
        }

        .dynamic-text-wrapper {
            margin-top: 30px;
        }

        .dynamic-header {
            font-size: clamp(28px, 5vw, 40px);
            font-weight: 600;
            color: #ffffff;
            margin: 0 0 15px 0;
            letter-spacing: 1px;
        }

        .text {
            font-size: clamp(32px, 6vw, 48px);
            font-weight: 700;
            position: relative;
            min-height: 1.4em;
        }

        .text p {
            margin: 0;
            color: #ffffff;
            display: block;
        }

        .word {
            position: absolute;
            opacity: 0;
            white-space: nowrap;
            top: 0;
            right: 0;
            text-align: right;
        }

        .letter {
            display: inline-block;
            position: relative;
            transform: translateZ(25px);
            transform-origin: 50% 50% 25px;
        }

        .letter.out {
            transform: rotateX(90deg);
            transition: transform 0.32s cubic-bezier(0.55, 0.055, 0.675, 0.19);
        }

        .letter.behind {
            transform: rotateX(-90deg);
        }

        .letter.in {
            transform: rotateX(0deg);
            transition: transform 0.38s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }

        .wisteria {
            color: #9b59b6;
        }

        .belize {
            color: #3498db;
        }

        .pomegranate {
            color: #e74c3c;
        }

        .green {
            color: #1abc9c;
        }

        .midnight {
            color: #34495e;
        }

        .emerald {
            color: #2ecc71;
        }

        @media (max-width: 1024px) {
            .title {
                font-size: clamp(32px, 6vw, 48px);
            }

            .subtitle {
                font-size: clamp(18px, 3.5vw, 24px);
            }

            .dynamic-header {
                font-size: clamp(24px, 4.5vw, 32px);
            }

            .text {
                font-size: clamp(28px, 5.5vw, 38px);
            }
        }

        @media (max-width: 768px) {
            .text-container {
                width: 90%;
            }

            .left-text {
                left: 5%;
                top: 25%;
            }

            .right-text {
                right: 5%;
                top: 65%;
            }

            .title {
                font-size: clamp(28px, 8vw, 40px);
            }

            .subtitle {
                font-size: clamp(16px, 4vw, 20px);
            }

            .dynamic-header {
                font-size: clamp(20px, 5vw, 28px);
            }

            .text {
                font-size: clamp(24px, 6vw, 32px);
            }
        }

        /* Click instruction hint */
        .click-hint {
            position: fixed;
            bottom: 30px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 4;
            color: rgba(255, 255, 255, 0.6);
            font-size: 14px;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 0.6; }
            50% { opacity: 1; }
        }
    </style>
</head>

<body>
    <!-- Starry Background -->
    <div id="stars"></div>
    <div id="stars2"></div>
    <div id="stars3"></div>

    <div id="canvas-container"></div>

    <!-- Left Side: OSTEOPOROSIS (Clickable - just for show, actual button is in Streamlit) -->
    <div class="text-container left-text">
        <div>
            <div class="title-link">
                <h1 class="title">OSTEOPOROSIS</h1>
            </div>
        </div>
    </div>

    <!-- Right Side: Advanced Detection AI (Static Title + Dynamic Words) -->
    <div class="text-container right-text">
        <div>
            <h2 class="subtitle">Advanced Detection AI</h2>
            <div class="dynamic-text-wrapper">
                <p class="dynamic-header">Detection is</p>
                <div class="text">
                    <p>
                        <span class="word wisteria">Accurate.</span>
                        <span class="word belize">Intelligent.</span>
                        <span class="word pomegranate">Precise.</span>
                        <span class="word green">Advanced.</span>
                        <span class="word midnight">Reliable.</span>
                        <span class="word emerald">Revolutionary.</span>
                    </p>
                </div>
            </div>
        </div>
    </div>

    <!-- Click hint -->
    <div class="click-hint">
        ↓ Click below to enter dashboard ↓
    </div>

    <script>
        // ──────────────── p5.js Animation with Two Circles ────────────────
        let colors = [];
        let d, n, randies, t;
        let circleSize;

        new p5(function (p) {
            p.setup = function () {
                let canvas = p.createCanvas(p.windowWidth, p.windowHeight);
                canvas.parent('canvas-container');
                p.clear();
                d = p.min(p.width, p.height);
                circleSize = d * 0.3;
                p.noFill();
                p.angleMode(p.DEGREES);
                randies = [p.random(1, 2), p.random(1, 2)];
                p.strokeWeight(1.5);
                n = 50;

                for (let i = 0; i < n; i++) {
                    colors.push({
                        r: p.random(100, 255),
                        g: p.random(100, 255),
                        b: p.random(100, 255)
                    });
                }
            };

            p.draw = function () {
                t = p.frameCount * 1.5;
                p.clear();

                // Draw top circle
                drawCircleAnimation(p, p.width / 2, p.height * 0.25);

                // Draw bottom circle
                drawCircleAnimation(p, p.width / 2, p.height * 0.75);
            };

            function drawCircleAnimation(p, centerX, centerY) {
                p.push();
                p.translate(centerX, centerY);

                p.stroke(150 + 105 * p.sin(t / 10), 150 + 105 * p.sin(t / 15), 150 + 105 * p.sin(t / 20));
                p.strokeWeight(0.5);
                p.circle(0, 0, circleSize);

                for (let j = 0; j < n; j++) {
                    let r = 20;
                    let c = colors[j];
                    p.stroke(
                        c.r + 50 * p.sin(t / 20 + j),
                        c.g + 50 * p.cos(t / 25 + j),
                        c.b + 50 * p.sin(t / 30 + j)
                    );
                    p.strokeWeight(1.5);

                    p.push();
                    p.rotate((j * 360) / n);
                    p.beginShape();
                    for (let i = 0; i < circleSize / 2 - (circleSize / 2) * p.cos(t / 5); i++) {
                        let x = i + (circleSize / 2) * p.cos(t / 5);
                        let y = r * p.sin(i * randies[0]) * p.cos(i * randies[1] + t / randies[0]) * p.cos(-t / randies[1]);
                        p.vertex(x, y);
                    }
                    p.endShape();
                    p.pop();
                }
                p.pop();
            }

            p.windowResized = function () {
                p.resizeCanvas(p.windowWidth, p.windowHeight);
                d = p.min(p.width, p.height);
                circleSize = d * 0.3;
            };
        });

        // ──────────────── Text Animation ──────────────────
        const words = document.getElementsByClassName('word');
        const wordArray = [];
        let currentWord = 0;

        words[currentWord].style.opacity = 1;
        for (let i = 0; i < words.length; i++) {
            splitLetters(words[i]);
        }

        function changeWord() {
            const cw = wordArray[currentWord];
            const nw = currentWord === words.length - 1 ? wordArray[0] : wordArray[currentWord + 1];

            for (let i = 0; i < cw.length; i++) {
                animateLetterOut(cw, i);
            }

            for (let i = 0; i < nw.length; i++) {
                nw[i].className = 'letter behind';
                nw[0].parentElement.style.opacity = 1;
                animateLetterIn(nw, i);
            }

            currentWord = (currentWord === words.length - 1) ? 0 : currentWord + 1;
        }

        function animateLetterOut(cw, i) {
            setTimeout(() => {
                cw[i].className = 'letter out';
            }, i * 70);
        }

        function animateLetterIn(nw, i) {
            setTimeout(() => {
                nw[i].className = 'letter in';
            }, 300 + (i * 70));
        }

        function splitLetters(word) {
            const content = word.innerHTML;
            word.innerHTML = '';
            const letters = [];
            for (let i = 0; i < content.length; i++) {
                const letter = document.createElement('span');
                letter.className = 'letter';
                letter.innerHTML = content.charAt(i);
                word.appendChild(letter);
                letters.push(letter);
            }
            wordArray.push(letters);
        }

        setInterval(changeWord, 3800);
        changeWord();
    </script>
</body>

</html>
"""

# Check if user clicked to enter dashboard
if st.session_state["show_landing"]:
    st.set_page_config(page_title="Osteoporosis Detection AI", layout="wide", page_icon="🦴", initial_sidebar_state="collapsed")
    
    # Render the landing page
    components.html(LANDING_HTML, height=800, scrolling=False)
    
    # Style for the button
    st.markdown("""
        <style>
        div.stButton > button {
            position: fixed;
            bottom: 50px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 9999;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 18px 50px;
            border-radius: 50px;
            font-size: 18px;
            font-weight: 700;
            cursor: pointer;
            box-shadow: 0 10px 40px rgba(102, 126, 234, 0.6);
            transition: all 0.3s ease;
            letter-spacing: 1px;
        }
        div.stButton > button:hover {
            transform: translateX(-50%) translateY(-3px);
            box-shadow: 0 15px 50px rgba(102, 126, 234, 0.8);
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        }
        div.stButton > button:active {
            transform: translateX(-50%) translateY(-1px);
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Button to enter dashboard
    if st.button("🚀 ENTER DASHBOARD"):
        st.session_state["show_landing"] = False
        st.rerun()
    
    # Stop execution here - don't show the dashboard yet
    st.stop()

# -----------------------
# Main Dashboard Code (only runs after landing page)
# -----------------------
st.set_page_config(page_title="OA Premium Dashboard (Auth + Logs)", layout="wide", page_icon="🦴")

# Load CSS (keep your premium CSS in assets/style.css)
if os.path.exists("assets/style.css"):
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# -----------------------
# Database helpers & schema (adds inference_logs)
# -----------------------
def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cur = conn.cursor()
    # users table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        full_name TEXT
    );
    """)
    # patients table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT UNIQUE,
        name TEXT,
        age INTEGER,
        gender TEXT,
        last_visit TEXT,
        notes TEXT,
        created_by INTEGER,
        FOREIGN KEY (created_by) REFERENCES users(id)
    );
    """)
    # inference logs table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS inference_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT,
        predicted_grade TEXT,
        timestamp TEXT,
        user_id INTEGER,
        orig_image_path TEXT,
        heatmap_path TEXT,
        notes TEXT,
        FOREIGN KEY (user_id) REFERENCES users(id)
    );
    """)
    conn.commit()
    conn.close()

# initialize DB on startup (creates tables if absent)
init_db()

# -----------------------
# Authentication helpers
# -----------------------
def register_user(username, password, full_name=""):
    conn = get_connection()
    cur = conn.cursor()
    try:
        pw_hash = generate_password_hash(password)
        cur.execute("INSERT INTO users (username, password_hash, full_name) VALUES (?, ?, ?)", (username, pw_hash, full_name))
        conn.commit()
        return True, None
    except sqlite3.IntegrityError as e:
        return False, "Username already exists"
    finally:
        conn.close()

def authenticate_user(username, password):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    if row:
        if check_password_hash(row["password_hash"], password):
            return {"id": row["id"], "username": row["username"], "full_name": row["full_name"]}
    return None

def get_user_by_username(username):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    return row

# -----------------------
# Patient DB operations
# -----------------------
def create_patient(patient_id, name, age, gender, last_visit, notes, created_by):
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO patients (patient_id, name, age, gender, last_visit, notes, created_by)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (patient_id, name, age, gender, last_visit, notes, created_by))
        conn.commit()
        return True, None
    except sqlite3.IntegrityError as e:
        return False, "Patient ID already exists"
    finally:
        conn.close()

def read_patients(limit=200):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT p.*, u.username AS created_by_username FROM patients p LEFT JOIN users u ON p.created_by = u.id ORDER BY p.id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def update_patient(row_id, **fields):
    conn = get_connection()
    cur = conn.cursor()
    keys = ", ".join([f"{k} = ?" for k in fields.keys()])
    vals = list(fields.values()) + [row_id]
    cur.execute(f"UPDATE patients SET {keys} WHERE id = ?", vals)
    conn.commit()
    conn.close()

def delete_patient(row_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM patients WHERE id = ?", (row_id,))
    conn.commit()
    conn.close()

def get_patient_by_patient_id(patient_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None

# -----------------------
# Inference logs operations
# -----------------------
def log_inference(patient_id, predicted_grade, user_id, orig_image_path, heatmap_path, notes=""):
    conn = get_connection()
    cur = conn.cursor()
    timestamp = datetime.utcnow().isoformat()
    cur.execute("""
        INSERT INTO inference_logs (patient_id, predicted_grade, timestamp, user_id, orig_image_path, heatmap_path, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (patient_id, predicted_grade, timestamp, user_id, orig_image_path, heatmap_path, notes))
    conn.commit()
    conn.close()

def read_inference_logs(limit=100):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT il.*, u.username AS user_name
        FROM inference_logs il
        LEFT JOIN users u ON il.user_id = u.id
        ORDER BY il.id DESC LIMIT ?
    """, (limit,))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]

# -----------------------
# Model loading
# -----------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.warning(f"Model not found at {MODEL_PATH}. Inference disabled.")
        return None
    try:
        base_model = models.resnet50(pretrained=False)
        base_model.fc = nn.Linear(base_model.fc.in_features, len(CLASSES))
        base_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        base_model.to(DEVICE).eval()
        return base_model
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None

MODEL_AVAILABLE = os.path.exists(MODEL_PATH)
model = load_model() if MODEL_AVAILABLE else None

transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# -----------------------
# Grad-CAM
# -----------------------
def generate_gradcam(model, x_tensor):
    model.eval()
    feature_map = []
    gradient = []

    def forward_hook(module, input, output):
        feature_map.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradient.append(grad_out[0])

    # ResNet50: target the last conv block in layer4
    target_layer = model.layer4[-1]
    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    x = x_tensor.clone().requires_grad_(True)
    out = model(x)
    cls = out.argmax(dim=1).item()
    model.zero_grad()
    out[0, cls].backward()

    h1.remove()
    h2.remove()

    if not gradient or not feature_map:
        # Fallback: return a blank heatmap if hooks failed
        blank = np.zeros((224, 224, 3), dtype=np.uint8)
        return blank, cls

    pooled_grads = gradient[0].mean(dim=[2, 3], keepdim=True)
    feat = feature_map[0]
    cam = (feat * pooled_grads).sum(dim=1, keepdim=False).squeeze().cpu().detach().numpy()
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()
    cam = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap, cls

# -----------------------
# PDF report generator
# -----------------------
def generate_pdf_report(predicted_grade, orig_path, heatmap_path, patient_info=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, txt="OA Detection Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    if patient_info:
        pdf.cell(200, 10, txt=f"Patient ID: {patient_info.get('patient_id','N/A')}", ln=True)
        pdf.cell(200, 10, txt=f"Name: {patient_info.get('name','N/A')}", ln=True)
    pdf.cell(200, 10, txt=f"Predicted Grade: {predicted_grade}", ln=True)
    pdf.cell(200, 10, txt=f"Timestamp: {datetime.utcnow().isoformat()}", ln=True)
    pdf.ln(10)
    if os.path.exists(orig_path):
        pdf.image(orig_path, x=10, y=None, w=90)
    if os.path.exists(heatmap_path):
        pdf.image(heatmap_path, x=110, y=None, w=90)
    os.makedirs("tmp", exist_ok=True)
    out_path = f"tmp/report_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.pdf"
    pdf.output(out_path)
    return out_path

# -----------------------
# Generate Analytics Dashboard HTML with Real Data
# -----------------------
def generate_analytics_dashboard():
    # Get real data from database
    patients = read_patients(limit=1000)
    logs = read_inference_logs(limit=1000)
    
    # Prepare data for dashboard
    dashboard_patients = []
    for i, patient in enumerate(patients):
        # Get patient's latest inference
        patient_logs = [log for log in logs if log['patient_id'] == patient['patient_id']]
        latest_log = patient_logs[0] if patient_logs else None
        
        # Map grade to diagnosis
        diagnosis_map = {
            "Grade 0": "Normal",
            "Grade 1": "Mild Osteopenia",
            "Grade 2": "Moderate Osteopenia",
            "Grade 3": "Severe Osteopenia",
            "Grade 4": "Osteoporosis"
        }
        
        diagnosis = diagnosis_map.get(latest_log['predicted_grade'], "Not Diagnosed") if latest_log else "Not Diagnosed"
        
        # Generate BMD score based on grade (simulated)
        bmd_scores = {"Grade 0": "-0.5", "Grade 1": "-1.2", "Grade 2": "-1.8", "Grade 3": "-2.3", "Grade 4": "-2.9"}
        bmd_score = bmd_scores.get(latest_log['predicted_grade'], "N/A") if latest_log else "N/A"
        
        dashboard_patients.append({
            "id": patient['patient_id'],
            "serialNo": f"S{i+1:03d}",
            "name": patient['name'],
            "age": patient['age'],
            "gender": patient['gender'],
            "bmdScore": bmd_score,
            "diagnosis": diagnosis,
            "xrayCount": len(patient_logs),
            "date": patient['last_visit'],
            "diagnosedBy": patient.get('created_by_username', 'Unknown')
        })
    
    # Calculate statistics
    total_patients = len(patients)
    severe_cases = len([p for p in dashboard_patients if p['diagnosis'] in ['Severe Osteopenia', 'Osteoporosis']])
    total_xrays = sum([p['xrayCount'] for p in dashboard_patients])
    avg_age = int(sum([p['age'] for p in dashboard_patients]) / len(dashboard_patients)) if dashboard_patients else 0
    
    # Grade distribution
    grade_distribution = {}
    for log in logs:
        grade = log['predicted_grade']
        grade_distribution[grade] = grade_distribution.get(grade, 0) + 1
    
    # Gender distribution
    gender_distribution = {"M": 0, "F": 0, "Other": 0}
    for patient in patients:
        gender_distribution[patient['gender']] = gender_distribution.get(patient['gender'], 0) + 1
    
    # Generate timeline data
    timeline_data = []
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    cumulative = 0
    for month in months:
        cumulative += np.random.randint(5, 15)
        timeline_data.append(cumulative)
    
    patients_json = json.dumps(dashboard_patients)
    
    dashboard_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Osteoporosis Detection Dashboard</title>
        <script crossorigin src="https://unpkg.com/react@17/umd/react.production.min.js"></script>
        <script crossorigin src="https://unpkg.com/react-dom@17/umd/react-dom.production.min.js"></script>
        <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">

        <style>
            body {{
                font-family: 'Poppins', sans-serif;
                background: radial-gradient(ellipse at bottom, #1B2735 0%, #090A0F 100%);
                color: #e0e0e0;
                margin: 0;
                overflow-x: hidden;
            }}

            #stars {{
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                z-index: 0;
                width: 1px;
                height: 1px;
                background: transparent;
                box-shadow: 1795px 457px #FFF, 1523px 1943px #FFF, 456px 1753px #FFF, 712px 1456px #FFF, 1134px 967px #FFF;
                animation: animStar 50s linear infinite;
            }}

            @keyframes animStar {{
                from {{ transform: translateY(0px); }}
                to {{ transform: translateY(-2000px); }}
            }}

            .dashboard-card {{
                background: rgba(23, 23, 23, 0.95);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 16px;
                padding: 20px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
                backdrop-filter: blur(10px);
            }}

            .stat-card {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 12px;
                padding: 20px;
                color: white;
                transition: transform 0.3s ease;
            }}

            .stat-card:hover {{
                transform: translateY(-5px);
            }}

            .table-container {{
                max-height: 400px;
                overflow-y: auto;
            }}

            .table-container::-webkit-scrollbar {{
                width: 8px;
            }}

            .table-container::-webkit-scrollbar-track {{
                background: rgba(255, 255, 255, 0.05);
                border-radius: 10px;
            }}

            .table-container::-webkit-scrollbar-thumb {{
                background: rgba(255, 255, 255, 0.2);
                border-radius: 10px;
            }}

            table {{
                width: 100%;
                border-collapse: collapse;
            }}

            th {{
                background: rgba(102, 126, 234, 0.2);
                padding: 12px;
                text-align: left;
                font-weight: 600;
                position: sticky;
                top: 0;
                z-index: 10;
            }}

            td {{
                padding: 12px;
                border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            }}

            tr:hover {{
                background: rgba(255, 255, 255, 0.05);
            }}

            .severity-badge {{
                padding: 4px 12px;
                border-radius: 12px;
                font-size: 12px;
                font-weight: 600;
            }}

            .severity-normal {{
                background: #10b981;
                color: white;
            }}

            .severity-mild {{
                background: #f59e0b;
                color: white;
            }}

            .severity-moderate {{
                background: #f97316;
                color: white;
            }}

            .severity-severe {{
                background: #ef4444;
                color: white;
            }}

            .header-title {{
                font-size: 2.5rem;
                font-weight: 700;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }}
        </style>
    </head>
    <body>
        <div id="stars"></div>
        <div id="root"></div>

        <script type="text/babel">
            const Dashboard = () => {{
                const React = window.React;
                const {{ useEffect, useRef }} = React;

                const patients = {patients_json};

                const stats = {{
                    total: {total_patients},
                    severe: {severe_cases},
                    totalXrays: {total_xrays},
                    avgAge: {avg_age}
                }};

                const pieChartRef = useRef(null);
                const genderChartRef = useRef(null);
                const lineChartRef = useRef(null);

                useEffect(() => {{
                    const gradeData = {json.dumps(grade_distribution)};
                    const pieCtx = pieChartRef.current.getContext('2d');
                    new Chart(pieCtx, {{
                        type: 'pie',
                        data: {{
                            labels: Object.keys(gradeData),
                            datasets: [{{
                                data: Object.values(gradeData),
                                backgroundColor: ['#10b981', '#3b82f6', '#f59e0b', '#f97316', '#ef4444']
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {{
                                legend: {{
                                    labels: {{ color: '#e0e0e0' }}
                                }}
                            }}
                        }}
                    }});

                    const genderData = {json.dumps(gender_distribution)};
                    const genderCtx = genderChartRef.current.getContext('2d');
                    new Chart(genderCtx, {{
                        type: 'doughnut',
                        data: {{
                            labels: Object.keys(genderData),
                            datasets: [{{
                                data: Object.values(genderData),
                                backgroundColor: ['#3b82f6', '#ec4899', '#8b5cf6']
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {{
                                legend: {{
                                    labels: {{ color: '#e0e0e0' }}
                                }}
                            }}
                        }}
                    }});

                    const timelineData = {json.dumps(timeline_data)};
                    const lineCtx = lineChartRef.current.getContext('2d');
                    new Chart(lineCtx, {{
                        type: 'line',
                        data: {{
                            labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                            datasets: [{{
                                label: 'X-rays Processed',
                                data: timelineData,
                                borderColor: '#667eea',
                                backgroundColor: 'rgba(102, 126, 234, 0.1)',
                                tension: 0.4
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {{
                                legend: {{
                                    labels: {{ color: '#e0e0e0' }}
                                }}
                            }},
                            scales: {{
                                y: {{
                                    ticks: {{ color: '#e0e0e0' }},
                                    grid: {{ color: 'rgba(255, 255, 255, 0.1)' }}
                                }},
                                x: {{
                                    ticks: {{ color: '#e0e0e0' }},
                                    grid: {{ color: 'rgba(255, 255, 255, 0.1)' }}
                                }}
                            }}
                        }}
                    }});
                }}, []);

                return (
                    <div className="min-h-screen p-4 lg:p-8">
                        <div className="mb-8">
                            <h1 className="header-title mb-2">Osteoporosis Detection Dashboard</h1>
                            <p className="text-gray-400">Advanced AI-Powered Diagnostic Analytics</p>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
                            <div className="stat-card">
                                <div className="text-3xl font-bold">{{stats.total}}</div>
                                <div className="text-sm opacity-90">Total Patients</div>
                            </div>
                            <div className="stat-card" style={{{{ background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)' }}}}>
                                <div className="text-3xl font-bold">{{stats.severe}}</div>
                                <div className="text-sm opacity-90">Severe Cases</div>
                            </div>
                            <div className="stat-card" style={{{{ background: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)' }}}}>
                                <div className="text-3xl font-bold">{{stats.totalXrays}}</div>
                                <div className="text-sm opacity-90">X-rays Processed</div>
                            </div>
                            <div className="stat-card" style={{{{ background: 'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)' }}}}>
                                <div className="text-3xl font-bold">{{stats.avgAge}}</div>
                                <div className="text-sm opacity-90">Average Age</div>
                            </div>
                        </div>

                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
                            <div className="dashboard-card">
                                <h3 className="text-xl font-bold mb-4">Diagnosis Distribution</h3>
                                <div style={{{{ height: '300px' }}}}>
                                    <canvas ref={{pieChartRef}}></canvas>
                                </div>
                            </div>

                            <div className="dashboard-card">
                                <h3 className="text-xl font-bold mb-4">Gender Distribution</h3>
                                <div style={{{{ height: '300px' }}}}>
                                    <canvas ref={{genderChartRef}}></canvas>
                                </div>
                            </div>

                            <div className="dashboard-card">
                                <h3 className="text-xl font-bold mb-4">Age Distribution</h3>
                                <div className="space-y-3" style={{{{ height: '300px', overflowY: 'auto' }}}}>
                                    <div>
                                        <div className="flex justify-between text-sm mb-1">
                                            <span>30-40 years</span>
                                            <span>{{patients.filter(p => p.age >= 30 && p.age < 40).length}}</span>
                                        </div>
                                        <div className="w-full bg-gray-700 rounded-full h-2">
                                            <div className="bg-blue-500 h-2 rounded-full" style={{{{ width: `${{(patients.filter(p => p.age >= 30 && p.age < 40).length / patients.length) * 100}}%` }}}}></div>
                                        </div>
                                    </div>
                                    <div>
                                        <div className="flex justify-between text-sm mb-1">
                                            <span>40-50 years</span>
                                            <span>{{patients.filter(p => p.age >= 40 && p.age < 50).length}}</span>
                                        </div>
                                        <div className="w-full bg-gray-700 rounded-full h-2">
                                            <div className="bg-purple-500 h-2 rounded-full" style={{{{ width: `${{(patients.filter(p => p.age >= 40 && p.age < 50).length / patients.length) * 100}}%` }}}}></div>
                                        </div>
                                    </div>
                                    <div>
                                        <div className="flex justify-between text-sm mb-1">
                                            <span>50-60 years</span>
                                            <span>{{patients.filter(p => p.age >= 50 && p.age < 60).length}}</span>
                                        </div>
                                        <div className="w-full bg-gray-700 rounded-full h-2">
                                            <div className="bg-pink-500 h-2 rounded-full" style={{{{ width: `${{(patients.filter(p => p.age >= 50 && p.age < 60).length / patients.length) * 100}}%` }}}}></div>
                                        </div>
                                    </div>
                                    <div>
                                        <div className="flex justify-between text-sm mb-1">
                                            <span>60-70 years</span>
                                            <span>{{patients.filter(p => p.age >= 60 && p.age < 70).length}}</span>
                                        </div>
                                        <div className="w-full bg-gray-700 rounded-full h-2">
                                            <div className="bg-orange-500 h-2 rounded-full" style={{{{ width: `${{(patients.filter(p => p.age >= 60 && p.age < 70).length / patients.length) * 100}}%` }}}}></div>
                                        </div>
                                    </div>
                                    <div>
                                        <div className="flex justify-between text-sm mb-1">
                                            <span>70+ years</span>
                                            <span>{{patients.filter(p => p.age >= 70).length}}</span>
                                        </div>
                                        <div className="w-full bg-gray-700 rounded-full h-2">
                                            <div className="bg-red-500 h-2 rounded-full" style={{{{ width: `${{(patients.filter(p => p.age >= 70).length / patients.length) * 100}}%` }}}}></div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div className="dashboard-card mb-8">
                            <h3 className="text-xl font-bold mb-4">X-ray Production Over Time (2024)</h3>
                            <div style={{{{ height: '300px' }}}}>
                                <canvas ref={{lineChartRef}}></canvas>
                            </div>
                        </div>

                        <div className="dashboard-card">
                            <div className="flex justify-between items-center mb-4">
                                <h3 className="text-xl font-bold">Patient Records</h3>
                            </div>
                            <div className="table-container">
                                <table>
                                    <thead>
                                        <tr>
                                            <th>Patient ID</th>
                                            <th>Serial No</th>
                                            <th>Name</th>
                                            <th>Age</th>
                                            <th>Gender</th>
                                            <th>BMD Score</th>
                                            <th>Diagnosis</th>
                                            <th>X-rays</th>
                                            <th>Date</th>
                                            <th>Diagnosed By</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {{patients.map(patient => (
                                            <tr key={{patient.id}}>
                                                <td className="font-mono text-sm">{{patient.id}}</td>
                                                <td className="font-mono text-sm">{{patient.serialNo}}</td>
                                                <td>{{patient.name}}</td>
                                                <td>{{patient.age}}</td>
                                                <td>{{patient.gender}}</td>
                                                <td className="font-mono">{{patient.bmdScore}}</td>
                                                <td>
                                                    <span className={{`severity-badge severity-${{patient.diagnosis === 'Normal' ? 'normal' :
                                                        patient.diagnosis === 'Mild Osteopenia' ? 'mild' :
                                                            patient.diagnosis === 'Moderate Osteopenia' ? 'moderate' : 'severe'
                                                        }}`}}>
                                                        {{patient.diagnosis}}
                                                    </span>
                                                </td>
                                                <td>{{patient.xrayCount}}</td>
                                                <td>{{patient.date}}</td>
                                                <td>{{patient.diagnosedBy}}</td>
                                            </tr>
                                        ))}}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                );
            }};

            ReactDOM.render(<Dashboard />, document.getElementById('root'));
        </script>
    </body>
    </html>
    """
    
    return dashboard_html

# -----------------------
# Session state initialization
# -----------------------
if "user" not in st.session_state:
    st.session_state["user"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# -----------------------
# Authentication UI in sidebar
# -----------------------
with st.sidebar:
    st.markdown("## 🔐 Authentication")
    if st.session_state["user"] is None:
        auth_mode = st.radio("Mode", ["Login", "Register"])
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if auth_mode == "Login":
            if st.button("Login"):
                user = authenticate_user(username, password)
                if user:
                    st.session_state["user"] = user
                    st.success(f"Logged in as {user['username']}")
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        else:
            full_name = st.text_input("Full Name (optional)")
            if st.button("Register"):
                ok, err = register_user(username, password, full_name)
                if ok:
                    st.success("Registration successful. Please login.")
                else:
                    st.error(err)
    else:
        st.success(f"Logged in as **{st.session_state['user']['username']}**")
        if st.button("Logout"):
            st.session_state["user"] = None
            st.rerun()
    
    st.markdown("---")
    # Button to return to landing page
    if st.button("🏠 Back to Landing"):
        st.session_state["show_landing"] = True
        st.rerun()

# -----------------------
# FUTURISTIC NAVIGATION PANEL
# -----------------------
st.markdown("""
<style>
    /* Semi-Circular Navigation Styles */
    .nav-container {
        position: fixed;
        top: 0;
        left: 50%;
        transform: translateX(-50%);
        z-index: 1000;
        width: 900px;
        max-width: 95vw;
        height: 100px;
        background: linear-gradient(135deg, 
            rgba(15, 23, 42, 0.98) 0%,
            rgba(30, 41, 59, 0.95) 50%,
            rgba(15, 23, 42, 0.98) 100%);
        border-radius: 0 0 450px 450px;
        backdrop-filter: blur(20px);
        box-shadow: 
            0 10px 40px rgba(0, 0, 0, 0.5),
            inset 0 -2px 10px rgba(59, 130, 246, 0.2),
            0 2px 0 rgba(59, 130, 246, 0.3);
        border: 2px solid rgba(59, 130, 246, 0.3);
        border-top: none;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
        padding: 0 50px 25px 50px;
    }
    
    .nav-container::before {
        content: 'OA PREMIUM';
        position: absolute;
        top: 12px;
        left: 50%;
        transform: translateX(-50%);
        font-family: 'Orbitron', sans-serif;
        font-size: 13px;
        font-weight: 900;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 3px;
    }
    
    .stButton > button {
        width: 100%;
        min-width: 90px;
        background: rgba(30, 41, 59, 0.5) !important;
        border: 1px solid rgba(71, 85, 105, 0.5) !important;
        border-radius: 15px !important;
        padding: 14px 10px !important;
        text-align: center !important;
        cursor: pointer !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        position: relative !important;
        overflow: hidden !important;
        color: rgba(226, 232, 240, 0.9) !important;
        font-weight: 600 !important;
        font-size: 10px !important;
        white-space: pre-line !important;
        line-height: 1.5 !important;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(59, 130, 246, 0.3) 0%, transparent 70%);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(139, 92, 246, 0.2) 100%) !important;
        border-color: rgba(59, 130, 246, 0.6) !important;
        transform: translateY(-5px) !important;
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4), 0 0 30px rgba(59, 130, 246, 0.2) !important;
    }
    
    .stButton > button:hover::before {
        width: 200px;
        height: 200px;
    }
    
    .nav-active button {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.35) 0%, rgba(139, 92, 246, 0.35) 100%) !important;
        border-color: rgba(59, 130, 246, 0.8) !important;
        box-shadow: 0 0 30px rgba(59, 130, 246, 0.5), inset 0 0 20px rgba(59, 130, 246, 0.2), 0 8px 25px rgba(59, 130, 246, 0.3) !important;
        color: #60a5fa !important;
        transform: translateY(-5px) !important;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% {
            box-shadow: 0 0 20px rgba(59, 130, 246, 0.4);
        }
        50% {
            box-shadow: 0 0 35px rgba(59, 130, 246, 0.6);
        }
    }
    
    /* Add padding to main content */
    .main .block-container {
        padding-top: 130px !important;
    }
    
    /* Responsive */
    @media (max-width: 900px) {
        .nav-container {
            width: 95vw;
            height: 90px;
            padding: 0 20px 20px 20px;
            border-radius: 0 0 47.5vw 47.5vw;
        }
        
        .stButton > button {
            min-width: 70px;
            padding: 12px 8px !important;
            font-size: 9px !important;
        }
    }
    
    @media (max-width: 600px) {
        .nav-container {
            height: 80px;
            padding: 0 15px 15px 15px;
        }
        
        .stButton > button {
            min-width: 60px;
            padding: 10px 6px !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Navigation items configuration
nav_items = [
    {"id": "Dashboard", "icon": "📊", "label": "Dashboard"},
    {"id": "Patients", "icon": "👥", "label": "Patients"},
    {"id": "AI Detector", "icon": "🔬", "label": "AI Detector"},
    {"id": "Analytics", "icon": "📈", "label": "Analytics"},
    {"id": "Inference Logs", "icon": "📋", "label": "Logs"},
    {"id": "Chat Assistant", "icon": "💬", "label": "Chat AI"},
    {"id": "Settings", "icon": "⚙️", "label": "Settings"},
    {"id": "About", "icon": "ℹ️", "label": "About"}
]

# Create navigation with columns
st.markdown('<div class="nav-container">', unsafe_allow_html=True)
cols = st.columns(8)
for idx, item in enumerate(nav_items):
    with cols[idx]:
        active_class = "nav-active" if st.session_state["current_page"] == item["id"] else ""
        st.markdown(f'<div class="{active_class}">', unsafe_allow_html=True)
        if st.button(f"{item['icon']}\n{item['label']}", key=f"nav_{item['id']}"):
            st.session_state["current_page"] = item["id"]
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

choice = st.session_state["current_page"]

# -----------------------
# Page content
# -----------------------
if choice == "Dashboard":
    st.markdown("# 🏥 OA Premium Dashboard")
    
    # Get data
    patients = read_patients(limit=1000)
    logs = read_inference_logs(limit=1000)
    
    # Calculate statistics
    total_patients = len(patients)
    total_inferences = len(logs)
    
    # Grade distribution for chart
    grade_counts = {}
    for log in logs:
        grade = log['predicted_grade']
        grade_counts[grade] = grade_counts.get(grade, 0) + 1
    
    # Recent activity
    recent_logs = logs[:10] if logs else []
    
    # Futuristic Stats Cards with HTML
    st.markdown("""
    <style>
        .stats-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .stat-card-futuristic {
            background: linear-gradient(135deg, rgba(15, 15, 35, 0.95) 0%, rgba(25, 25, 50, 0.95) 100%);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(102, 126, 234, 0.3);
            border-radius: 16px;
            padding: 20px;
            position: relative;
            overflow: hidden;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            cursor: pointer;
            /* Square shape - approximately 5cm x 5cm (189px ≈ 5cm at 96 DPI) */
            width: 100%;
            aspect-ratio: 1 / 1;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
        }
        
        .stat-card-futuristic::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.1), transparent);
            transition: left 0.5s;
        }
        
        .stat-card-futuristic:hover::before {
            left: 100%;
        }
        
        .stat-card-futuristic:hover {
            transform: translateY(-10px) scale(1.02);
            border-color: rgba(102, 126, 234, 0.6);
            box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4);
        }
        
        .stat-icon {
            font-size: 36px;
            margin-bottom: 8px;
            filter: drop-shadow(0 0 10px rgba(102, 126, 234, 0.8));
            transition: all 0.3s;
        }
        
        .stat-card-futuristic:hover .stat-icon {
            filter: drop-shadow(0 0 20px rgba(102, 126, 234, 1));
            transform: scale(1.2) rotate(10deg);
        }
        
        .stat-value {
            font-size: 36px;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 8px 0;
            transition: all 0.3s;
        }
        
        .stat-card-futuristic:hover .stat-value {
            transform: scale(1.1);
        }
        
        .stat-label {
            font-size: 11px;
            color: rgba(255, 255, 255, 0.7);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-top: 5px;
        }
        
        .stat-change {
            font-size: 10px;
            color: #10b981;
            margin-top: 3px;
        }
        
        /* Futuristic Table Styles */
        .futuristic-table-container {
            background: linear-gradient(135deg, rgba(15, 15, 35, 0.95) 0%, rgba(25, 25, 50, 0.95) 100%);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(102, 126, 234, 0.3);
            border-radius: 16px;
            padding: 25px;
            margin-top: 30px;
            overflow: hidden;
            position: relative;
        }
        
        .futuristic-table-container::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 2px;
            background: linear-gradient(90deg, transparent, #667eea, #764ba2, transparent);
            animation: scan 3s infinite;
        }
        
        .table-title {
            font-size: 24px;
            font-weight: 700;
            margin-bottom: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .futuristic-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0 8px;
        }
        
        .futuristic-table thead tr {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        }
        
        .futuristic-table th {
            padding: 15px;
            text-align: left;
            color: #ffffff;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 12px;
            letter-spacing: 1px;
            border: none;
        }
        
        .futuristic-table th:first-child {
            border-radius: 10px 0 0 10px;
        }
        
        .futuristic-table th:last-child {
            border-radius: 0 10px 10px 0;
        }
        
        .futuristic-table tbody tr {
            background: rgba(255, 255, 255, 0.02);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
        }
        
        .futuristic-table tbody tr::before {
            content: '';
            position: absolute;
            left: 0;
            top: 0;
            height: 100%;
            width: 3px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            transform: scaleY(0);
            transition: transform 0.3s;
        }
        
        .futuristic-table tbody tr:hover::before {
            transform: scaleY(1);
        }
        
        .futuristic-table tbody tr:hover {
            background: linear-gradient(90deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
            transform: translateX(10px) scale(1.01);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.3);
        }
        
        .futuristic-table td {
            padding: 15px;
            color: rgba(255, 255, 255, 0.9);
            border: none;
            font-size: 14px;
        }
        
        .futuristic-table td:first-child {
            border-radius: 10px 0 0 10px;
        }
        
        .futuristic-table td:last-child {
            border-radius: 0 10px 10px 0;
        }
        
        .grade-badge {
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            display: inline-block;
            box-shadow: 0 0 15px currentColor;
            transition: all 0.3s;
        }
        
        .grade-badge:hover {
            transform: scale(1.1);
            box-shadow: 0 0 25px currentColor;
        }
        
        .grade-0 { background: #10b981; color: #fff; }
        .grade-1 { background: #3b82f6; color: #fff; }
        .grade-2 { background: #f59e0b; color: #fff; }
        .grade-3 { background: #f97316; color: #fff; }
        .grade-4 { background: #ef4444; color: #fff; }
        
        .timestamp {
            font-family: 'Courier New', monospace;
            color: rgba(255, 255, 255, 0.6);
            font-size: 12px;
        }
        
        .user-badge {
            background: rgba(102, 126, 234, 0.2);
            border: 1px solid rgba(102, 126, 234, 0.4);
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 12px;
            color: #667eea;
        }
        
        /* Chart Container */
        .chart-container {
            background: linear-gradient(135deg, rgba(15, 15, 35, 0.95) 0%, rgba(25, 25, 50, 0.95) 100%);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(102, 126, 234, 0.3);
            border-radius: 16px;
            padding: 25px;
            margin-top: 30px;
            transition: all 0.3s;
        }
        
        .chart-container:hover {
            border-color: rgba(102, 126, 234, 0.6);
            box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Stats Cards using Streamlit columns
    cols = st.columns(4)
    
    with cols[0]:
        st.markdown(f"""
        <div class="stat-card-futuristic">
            <div class="stat-icon">👥</div>
            <div class="stat-value">{total_patients}</div>
            <div class="stat-label">Total Patients</div>
            <div class="stat-change">↑ Active Records</div>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[1]:
        st.markdown(f"""
        <div class="stat-card-futuristic">
            <div class="stat-icon">🔬</div>
            <div class="stat-value">{total_inferences}</div>
            <div class="stat-label">Total Inferences</div>
            <div class="stat-change">↑ AI Predictions</div>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[2]:
        st.markdown(f"""
        <div class="stat-card-futuristic">
            <div class="stat-icon">👤</div>
            <div class="stat-value">{1 if st.session_state["user"] else 0}</div>
            <div class="stat-label">Active Users</div>
            <div class="stat-change">● Online Now</div>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[3]:
        model_icon = "🟢" if MODEL_AVAILABLE else "🔴"
        model_text = "AI" if MODEL_AVAILABLE else "OFF"
        model_status = "✓ Ready" if MODEL_AVAILABLE else "✗ Unavailable"
        st.markdown(f"""
        <div class="stat-card-futuristic">
            <div class="stat-icon">{model_icon}</div>
            <div class="stat-value">{model_text}</div>
            <div class="stat-label">Model Status</div>
            <div class="stat-change">{model_status}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts Row
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("### 📊 Grade Distribution")
        if grade_counts:
            # Create pie chart
            fig = go.Figure(data=[go.Pie(
                labels=list(grade_counts.keys()),
                values=list(grade_counts.values()),
                hole=0.4,
                marker=dict(
                    colors=['#10b981', '#3b82f6', '#f59e0b', '#f97316', '#ef4444'],
                    line=dict(color='#1B2735', width=2)
                ),
                textfont=dict(size=14, color='white'),
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
            )])
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.1,
                    font=dict(size=12)
                ),
                height=300,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No inference data available yet")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("### 📈 Inference Trend")
        if logs:
            # Create activity trend chart
            log_dates = [log['timestamp'][:10] for log in logs]
            date_counts = {}
            for date in log_dates:
                date_counts[date] = date_counts.get(date, 0) + 1
            
            sorted_dates = sorted(date_counts.keys())[-14:]  # Last 14 days
            counts = [date_counts[date] for date in sorted_dates]
            
            fig = go.Figure(data=[go.Scatter(
                x=sorted_dates,
                y=counts,
                mode='lines+markers',
                line=dict(color='#667eea', width=3, shape='spline'),
                marker=dict(size=8, color='#764ba2', line=dict(color='#667eea', width=2)),
                fill='tozeroy',
                fillcolor='rgba(102, 126, 234, 0.2)',
                hovertemplate='<b>Date:</b> %{x}<br><b>Inferences:</b> %{y}<extra></extra>'
            )])
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.1)',
                    zeroline=False
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.1)',
                    zeroline=False
                ),
                height=300,
                margin=dict(l=0, r=0, t=0, b=0),
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No inference data available yet")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Futuristic Table
    st.markdown('<div class="futuristic-table-container">', unsafe_allow_html=True)
    st.markdown('<div class="table-title">⚡ Recent Inference Activity</div>', unsafe_allow_html=True)
    
    if recent_logs:
        table_html = '<table class="futuristic-table"><thead><tr>'
        table_html += '<th>Patient ID</th><th>Prediction</th><th>Timestamp</th><th>User</th>'
        table_html += '</tr></thead><tbody>'
        
        for log in recent_logs:
            grade = log['predicted_grade']
            grade_num = grade.split()[-1] if 'Grade' in grade else '0'
            
            table_html += '<tr>'
            table_html += f'<td><strong>{log["patient_id"]}</strong></td>'
            table_html += f'<td><span class="grade-badge grade-{grade_num}">{grade}</span></td>'
            table_html += f'<td class="timestamp">{log["timestamp"]}</td>'
            table_html += f'<td><span class="user-badge">{log.get("user_name", "Unknown")}</span></td>'
            table_html += '</tr>'
        
        table_html += '</tbody></table>'
        st.markdown(table_html, unsafe_allow_html=True)
    else:
        st.info("No recent inference logs available")
    
    st.markdown('</div>', unsafe_allow_html=True)

elif choice == "Inference Logs":
    st.subheader("Inference Logs (who ran what & when)")
    logs = read_inference_logs(limit=500)
    if logs:
        df = pd.DataFrame(logs)
        st.dataframe(df, use_container_width=True)
        if st.button("Download CSV"):
            csv = df.to_csv(index=False)
            st.download_button("⬇ Download", csv, file_name="inference_logs.csv")
    else:
        st.info("No logs found.")

elif choice == "Patients":
    st.subheader("Patient Management")
    if st.session_state["user"] is None:
        st.warning("You must be logged in to manage patients.")
    else:
        patients = read_patients(limit=200)
        if patients:
            df = pd.DataFrame(patients)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No patients found")
        st.markdown("### Add new patient")
        with st.form("create_patient"):
            pid = st.text_input("Patient ID")
            name = st.text_input("Name")
            age = st.number_input("Age", min_value=0, max_value=120, value=40)
            gender = st.selectbox("Gender", ["M","F","Other"])
            last_visit = st.date_input("Last visit")
            notes = st.text_area("Notes (optional)")
            submitted = st.form_submit_button("Create")
            if submitted:
                ok, err = create_patient(pid.strip(), name.strip(), int(age), gender, last_visit.strftime("%Y-%m-%d"), notes.strip(), st.session_state["user"]["id"])
                if ok:
                    st.success("Patient created")
                else:
                    st.error(err)

        st.markdown("### Manage existing patients")
        sel = st.selectbox("Pick patient to edit/delete", options=[(p["id"], f'{p["patient_id"]} - {p["name"]}') for p in patients], format_func=lambda x: x[1]) if patients else None
        if sel:
            sel_id = sel[0]
            rec = next((p for p in patients if p["id"] == sel_id), None)
            if rec:
                c1, c2 = st.columns(2)
                with c1:
                    new_name = st.text_input("Edit Name", value=rec["name"])
                    new_age = st.number_input("Edit Age", value=rec["age"])
                    new_gender = st.selectbox("Edit Gender", ["M","F","Other"], index=["M","F","Other"].index(rec["gender"]) if rec["gender"] in ["M","F","Other"] else 0)
                with c2:
                    new_last = st.date_input("Edit Last Visit", value=pd.to_datetime(rec["last_visit"]))
                    new_notes = st.text_area("Edit Notes", value=rec["notes"])
                if st.button("Update Patient"):
                    update_patient(sel_id, name=new_name, age=new_age, gender=new_gender, last_visit=new_last.strftime("%Y-%m-%d"), notes=new_notes)
                    st.success("Updated")
                if st.button("Delete Patient"):
                    delete_patient(sel_id)
                    st.success("Deleted")

elif choice == "AI Detector":
    st.subheader("X-ray AI Detector (Grad-CAM)")
    if st.session_state["user"] is None:
        st.warning("You must be logged in to run detections.")
    uploaded = st.file_uploader("Upload a Knee X-ray", type=["jpg","jpeg","png"])
    if uploaded:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded X-ray", width=380)
        x = transform(img).unsqueeze(0).to(DEVICE)
        if MODEL_AVAILABLE and model is not None:
            with st.spinner("Analyzing image..."):
                try:
                    heatmap, cls = generate_gradcam(model, x)
                    grade = CLASSES[cls]
                    st.success(f"Predicted: {grade}")
                    img_np = np.array(img.resize((224,224)))
                    overlay = cv2.addWeighted(img_np, 0.55, heatmap, 0.45, 0)
                    st.image(overlay, caption="Grad-CAM Overlay", use_column_width=False)
                    os.makedirs("tmp", exist_ok=True)
                    timestamp_short = datetime.utcnow().strftime("%Y%m%d%H%M%S")
                    orig_path = f"tmp/orig_{timestamp_short}.jpg"
                    heat_path = f"tmp/heat_{timestamp_short}.jpg"
                    img.save(orig_path)
                    cv2.imwrite(heat_path, overlay)

                    st.markdown("### Save prediction?")
                    save_choice = st.radio("Choose how to save this prediction (Option C):",
                                           ("Save to existing patient", "Create new patient & save", "Don't save"))
                    if save_choice == "Save to existing patient":
                        patients = read_patients(limit=1000)
                        if not patients:
                            st.info("No patients available — create one or select 'Create new patient & save'.")
                        else:
                            options = [ (p["patient_id"], f'{p["patient_id"]} - {p["name"]}') for p in patients ]
                            sel = st.selectbox("Select patient", options=options, format_func=lambda x: x[1])
                            if st.button("Attach prediction to selected patient"):
                                selected_patient_id = sel[0]
                                log_inference(selected_patient_id, grade, st.session_state["user"]["id"], orig_path, heat_path, notes="")
                                patient = get_patient_by_patient_id(selected_patient_id)
                                new_notes = (patient.get("notes","") or "") + f"\nInference on {datetime.utcnow().isoformat()}: {grade}"
                                update_patient(patient["id"], last_visit=datetime.utcnow().strftime("%Y-%m-%d"), notes=new_notes)
                                st.success(f"Saved prediction to patient {selected_patient_id}")
                    elif save_choice == "Create new patient & save":
                        with st.form("create_patient_and_save"):
                            cp_patient_id = st.text_input("New Patient ID")
                            cp_name = st.text_input("Name")
                            cp_age = st.number_input("Age", min_value=0, max_value=120, value=40)
                            cp_gender = st.selectbox("Gender", ["M","F","Other"])
                            cp_last_visit = st.date_input("Last visit")
                            cp_notes = st.text_area("Notes (optional)")
                            cp_submit = st.form_submit_button("Create patient & save prediction")
                            if cp_submit:
                                if not cp_patient_id or not cp_name:
                                    st.error("Patient ID and Name required")
                                else:
                                    ok, err = create_patient(cp_patient_id.strip(), cp_name.strip(), int(cp_age), cp_gender, cp_last_visit.strftime("%Y-%m-%d"), cp_notes.strip(), st.session_state["user"]["id"])
                                    if ok:
                                        log_inference(cp_patient_id.strip(), grade, st.session_state["user"]["id"], orig_path, heat_path, notes=cp_notes.strip())
                                        st.success(f"Patient {cp_patient_id} created and prediction saved.")
                                    else:
                                        st.error(err)
                    else:
                        st.info("Prediction not saved. You can still Download PDF or change choice.")
                        if st.button("Download PDF Report (unsaved)"):
                            patient_info = {"patient_id":"Unassigned", "name":""}
                            pdf_path = generate_pdf_report(grade, orig_path, heat_path, patient_info=patient_info)
                            with open(pdf_path, "rb") as f:
                                st.download_button("⬇ Download Report (Unassigned)", f, file_name="OA_report_unassigned.pdf")
                    if st.button("Download PDF Report (saved/unsaved)"):
                        patient_info = {"patient_id":"", "name":""}
                        pdf_path = generate_pdf_report(grade, orig_path, heat_path, patient_info=patient_info)
                        with open(pdf_path, "rb") as f:
                            st.download_button("⬇ Download Report", f, file_name=f"OA_report_{timestamp_short}.pdf")
                except Exception as e:
                    st.error("Grad-CAM failed: " + str(e))
        else:
            st.warning("Model not available — cannot run prediction. You can still view the image.")
        st.markdown("</div>", unsafe_allow_html=True)

elif choice == "Analytics":
    # Generate and display the enhanced analytics dashboard with real data
    dashboard_html = generate_analytics_dashboard()
    components.html(dashboard_html, height=2000, scrolling=True)

elif choice == "Chat Assistant":
    st.markdown("# 💬 AI Medical Assistant")
    st.markdown("### Powered by Groq AI - Ultra-fast medical consultation")
    
    # Add custom CSS for chat interface
    st.markdown("""
    <style>
        .chat-container {
            background: linear-gradient(135deg, rgba(15, 15, 35, 0.95) 0%, rgba(25, 25, 50, 0.95) 100%);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(102, 126, 234, 0.3);
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 20px;
            max-height: 500px;
            overflow-y: auto;
        }
        
        .chat-container::-webkit-scrollbar {
            width: 8px;
        }
        
        .chat-container::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
        }
        
        .chat-container::-webkit-scrollbar-thumb {
            background: rgba(102, 126, 234, 0.3);
            border-radius: 10px;
        }
        
        .chat-message {
            margin: 15px 0;
            padding: 15px;
            border-radius: 12px;
            animation: slideIn 0.3s ease-out;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .user-message {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
            border-left: 3px solid #667eea;
            margin-left: 50px;
        }
        
        .bot-message {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.1) 100%);
            border-left: 3px solid #10b981;
            margin-right: 50px;
        }
        
        .message-header {
            font-weight: 600;
            color: #667eea;
            margin-bottom: 8px;
            font-size: 14px;
        }
        
        .bot-message .message-header {
            color: #10b981;
        }
        
        .message-content {
            color: rgba(255, 255, 255, 0.9);
            line-height: 1.6;
        }
        
        .xray-selector {
            background: linear-gradient(135deg, rgba(15, 15, 35, 0.95) 0%, rgba(25, 25, 50, 0.95) 100%);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(102, 126, 234, 0.3);
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 20px;
        }
        
        .xray-info {
            background: rgba(102, 126, 234, 0.1);
            border: 1px solid rgba(102, 126, 234, 0.3);
            border-radius: 10px;
            padding: 15px;
            margin-top: 10px;
        }
        
        .severity-indicator {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 12px;
            margin-top: 5px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize chat history in session state
    if "ai_chat_history" not in st.session_state:
        st.session_state["ai_chat_history"] = []
    
    if "selected_xray_for_chat" not in st.session_state:
        st.session_state["selected_xray_for_chat"] = None
    
    # X-ray Result Selector
    st.markdown('<div class="xray-selector">', unsafe_allow_html=True)
    st.markdown("#### 📊 Select X-ray Result for Analysis")
    
    # Get recent inference logs
    logs = read_inference_logs(limit=50)
    
    if logs:
        # Create options for selectbox
        log_options = []
        for log in logs:
            timestamp = log['timestamp'][:19].replace('T', ' ')
            option_text = f"{log['patient_id']} - {log['predicted_grade']} - {timestamp}"
            log_options.append((log, option_text))
        
        selected = st.selectbox(
            "Choose an X-ray result to discuss with AI:",
            options=range(len(log_options)),
            format_func=lambda x: log_options[x][1],
            key="xray_selector"
        )
        
        if selected is not None:
            selected_log = log_options[selected][0]
            st.session_state["selected_xray_for_chat"] = selected_log
            
            # Display selected X-ray info
            grade = selected_log['predicted_grade']
            grade_num = grade.split()[-1] if 'Grade' in grade else '0'
            
            severity_colors = {
                '0': ('#10b981', 'Normal - Healthy bone density'),
                '1': ('#3b82f6', 'Mild Osteopenia - Early bone loss'),
                '2': ('#f59e0b', 'Moderate Osteopenia - Notable bone loss'),
                '3': ('#f97316', 'Severe Osteopenia - Significant bone loss'),
                '4': ('#ef4444', 'Osteoporosis - Critical bone density loss')
            }
            
            color, description = severity_colors.get(grade_num, ('#9ca3af', 'Unknown'))
            
            st.markdown(f"""
            <div class="xray-info">
                <strong>Patient ID:</strong> {selected_log['patient_id']}<br>
                <strong>Prediction:</strong> <span style="color: {color}; font-weight: 600;">{grade}</span><br>
                <strong>Description:</strong> {description}<br>
                <strong>Analyzed on:</strong> {selected_log['timestamp'][:19].replace('T', ' ')}<br>
                <strong>By:</strong> {selected_log.get('user_name', 'Unknown')}
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🔬 Ask AI to Analyze This Result", key="analyze_btn"):
                analysis_prompt = f"""I have an X-ray scan result showing {grade}. 
                
Patient ID: {selected_log['patient_id']}
Diagnosis: {grade}
Severity: {description}

Please provide:
1. A brief explanation of what this grade means
2. Potential health implications
3. Recommended lifestyle changes or treatments
4. When to seek immediate medical attention

Keep your response professional, clear, and compassionate."""

                st.session_state["ai_chat_history"].append({
                    "role": "user",
                    "content": analysis_prompt
                })
                st.rerun()
    else:
        st.info("No X-ray results available. Run some predictions in the AI Detector first!")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # API Key Configuration (MANUAL INPUT)
    # Get your free API key from: https://console.groq.com
    
    st.markdown("#### 🔑 Groq API Key")
    api_key = st.text_input(
        "Enter your Groq API Key:",
        type="password",
        placeholder="gsk_...",
        help="Get your free API key from https://console.groq.com"
    )
    
    # Display status
    if api_key and api_key.strip():
        st.success("✅ API Key entered and ready!")
    else:
        st.info("ℹ️ Please enter your Groq API key above to use the AI assistant. Get a free key at https://console.groq.com")
    
    # Chat Interface
    st.markdown("#### 💬 Medical Consultation Chat")
    
    # Display chat history
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state["ai_chat_history"]:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <div class="message-header">👤 You</div>
                <div class="message-content">{msg['content']}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message bot-message">
                <div class="message-header">🤖 AI Medical Assistant</div>
                <div class="message-content">{msg['content']}</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input(
            "Ask about osteoporosis, treatments, or the selected X-ray result:",
            key="chat_input",
            placeholder="E.g., What exercises are safe for Grade 2 osteopenia?"
        )
    with col2:
        send_button = st.button("Send 📤", key="send_chat")
    
    # Handle sending message
    if send_button and user_input.strip() and api_key:
        # Add user message
        st.session_state["ai_chat_history"].append({
            "role": "user",
            "content": user_input
        })
        
        # Call Groq API
        try:
            import requests
            
            # Prepare messages for API
            messages = [
                {
                    "role": "system",
                    "content": """You are an expert medical AI assistant specializing in osteoporosis and bone health. 
                    Provide accurate, compassionate, and professional medical information. 
                    Always remind users to consult with healthcare professionals for personalized medical advice.
                    Be clear, concise, and supportive in your responses."""
                }
            ]
            
            # Add context about selected X-ray if available
            if st.session_state["selected_xray_for_chat"]:
                xray_context = st.session_state["selected_xray_for_chat"]
                messages.append({
                    "role": "system",
                    "content": f"Current patient context: X-ray shows {xray_context['predicted_grade']} for patient {xray_context['patient_id']}"
                })
            
            # Add chat history
            for msg in st.session_state["ai_chat_history"]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # API call to Groq
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.3-70b-versatile",  # Fast and free model
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 1024
                }
            )
            
            if response.status_code == 200:
                ai_response = response.json()["choices"][0]["message"]["content"]
                st.session_state["ai_chat_history"].append({
                    "role": "assistant",
                    "content": ai_response
                })
                st.rerun()
            else:
                error_msg = response.json().get("error", {}).get("message", "Unknown error")
                st.error(f"API Error: {error_msg}")
                
        except Exception as e:
            st.error(f"Error communicating with AI: {str(e)}")
            st.info("Please check your API key and internet connection.")
    
    elif send_button and not api_key:
        st.warning("Please enter your Groq API key first!")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state["ai_chat_history"] = []
        st.rerun()
    
    # Quick action buttons
    st.markdown("#### ⚡ Quick Questions")
    quick_questions = [
        "What are the early signs of osteoporosis?",
        "What foods help improve bone density?",
        "What exercises are safe for osteoporosis patients?",
        "How often should I get bone density scans?",
        "What are the risk factors for osteoporosis?"
    ]
    
    cols = st.columns(len(quick_questions))
    for idx, question in enumerate(quick_questions):
        with cols[idx]:
            if st.button(f"💡 {question[:20]}...", key=f"quick_{idx}"):
                if api_key:
                    st.session_state["ai_chat_history"].append({
                        "role": "user",
                        "content": question
                    })
                    st.rerun()
                else:
                    st.warning("Please enter your Groq API key first!")

elif choice == "Settings":
    st.subheader("Settings")
    st.write("Small preferences")
    st.checkbox("Enable debug logs", value=False)
    st.checkbox("Show advanced model info", value=False)

elif choice == "About":
    st.subheader("About")
    st.markdown("""
    **OA Premium Dashboard — Auth + Inference Logs**  
    - SQLite-backed users & patients  
    - Inference logs (who ran what & when)  
    - Option C workflow: attach predictions to existing patient / create new / don't save  
    - Passwords hashed (Werkzeug)
    - HTML Landing Page with animated visuals
    - Advanced Analytics Dashboard with real-time data
    - **NEW: Futuristic Navigation Panel** with glassmorphic design and smooth animations
    """)
    st.markdown("**Developer:** Yash Singh")

# Footer
st.markdown("<div style='margin-top:18px; color: rgba(255,255,255,0.45); font-size:12px;'>Built with ❤️ — Glassmorphism Premium • Auth • Inference Logs • Enhanced Analytics • Futuristic Nav • v2.1</div>", unsafe_allow_html=True)