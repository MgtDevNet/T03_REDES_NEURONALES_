"""Compila la aplicacion web autocontenida app_pronostico_flota.html (sin servidor ni CDN).

Incluye:
- Mapa interactivo de Colombia (SVG puro, sin WebGL) donde hacer clic en 2 municipios
  (Origen -> Destino) selecciona automaticamente la ruta y actualiza pronostico, buses y conductores.
- Aclaracion explicita entre el modelo Especifico (por ruta) y el modelo General (global),
  permitiendo comparar ambos en la grafica.
- Buscador de texto para las 848 rutas + controles operativos en vivo + exportacion CSV.
"""

from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd

from eda_utils import DIVIPOLA_CSV
from flota import estimar_tiempo_ciclo_horas

MODULE_DIR = Path(__file__).resolve().parents[1]
RUTAS_CSV = MODULE_DIR / "data" / "processed" / "rutas.csv"
CAL_CSV = MODULE_DIR / "data" / "processed" / "calendario_diario.csv"
PRON_CSV = MODULE_DIR / "results" / "pronosticos_30d.csv"
MET_RUTA_CSV = MODULE_DIR / "results" / "metricas_por_ruta.csv"
OUT_HTML = MODULE_DIR / "app_pronostico_flota.html"

# Poligono simplificado del contorno continental de Colombia (lon, lat) para fondo SVG 100% offline
COLOMBIA_BORDER_LONLAT: list[tuple[float, float]] = [
    (-71.67, 12.44), (-71.15, 12.22), (-71.33, 11.85), (-72.00, 11.45), (-72.88, 11.10),
    (-73.30, 10.45), (-72.95, 9.75), (-73.30, 9.10), (-72.60, 8.65), (-72.35, 7.90),
    (-72.00, 7.05), (-70.65, 7.02), (-69.40, 6.20), (-67.45, 6.18), (-67.85, 5.10),
    (-67.80, 4.00), (-67.35, 3.15), (-67.05, 2.00), (-66.85, 1.22), (-68.10, 1.15),
    (-69.85, 0.55), (-69.45, -1.10), (-69.45, -2.25), (-69.95, -4.22), (-70.35, -3.75),
    (-71.50, -2.55), (-73.05, -2.20), (-74.60, -0.55), (-75.25, -0.10), (-76.30, 0.35),
    (-77.55, 0.82), (-78.85, 1.45), (-78.65, 2.15), (-77.85, 2.70), (-77.20, 3.45),
    (-77.40, 4.25), (-77.35, 5.40), (-77.65, 6.25), (-77.85, 7.20), (-77.35, 8.65),
    (-76.75, 8.10), (-76.30, 8.85), (-75.60, 9.45), (-75.55, 10.40), (-74.85, 11.05),
    (-74.15, 11.25), (-73.15, 11.35), (-72.30, 11.80), (-71.67, 12.44),
]


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Pronóstico de Demanda, Mapa de Colombia y Flota — Módulo 1</title>
<style>
  :root {
    --bg: #0f172a;
    --panel: #1e293b;
    --panel-2: #263449;
    --border: #334155;
    --text: #f1f5f9;
    --muted: #94a3b8;
    --accent: #38bdf8;
    --accent-2: #f59e0b;
    --green: #22c55e;
    --red: #ef4444;
    --purple: #a78bfa;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.45;
  }
  header {
    background: linear-gradient(135deg, #0b1220 0%, #172554 100%);
    border-bottom: 1px solid var(--border);
    padding: 16px 26px;
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
  }
  header h1 { margin: 0; font-size: 1.32rem; letter-spacing: -0.01em; }
  header .sub { color: var(--muted); font-size: 0.85rem; margin-top: 3px; }
  .badges { display: flex; flex-wrap: wrap; gap: 8px; }
  .badge {
    background: rgba(56, 189, 248, 0.15);
    color: var(--accent);
    border: 1px solid rgba(56, 189, 248, 0.35);
    padding: 5px 11px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
  }
  .badge.esp {
    background: rgba(34, 197, 94, 0.15);
    color: #4ade80;
    border-color: rgba(34, 197, 94, 0.4);
  }
  .layout {
    display: grid;
    grid-template-columns: 350px 1fr;
    gap: 18px;
    padding: 18px 22px 36px;
    max-width: 1600px;
    margin: 0 auto;
  }
  @media (max-width: 1020px) {
    .layout { grid-template-columns: 1fr; }
  }
  .card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 15px 17px;
  }
  .sidebar { display: flex; flex-direction: column; gap: 14px; }
  .section-title {
    font-size: 0.76rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--accent);
    font-weight: 700;
    margin: 0 0 8px 0;
  }
  label {
    display: block;
    font-size: 0.82rem;
    color: var(--muted);
    margin-bottom: 4px;
    margin-top: 9px;
  }
  label strong { color: var(--text); float: right; font-weight: 600; }
  input[type="text"], input[type="number"], select {
    width: 100%;
    background: var(--bg);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 7px 10px;
    font-size: 0.86rem;
  }
  input[type="range"] { width: 100%; accent-color: var(--accent); cursor: pointer; }
  .route-meta {
    margin-top: 9px;
    padding: 9px 10px;
    background: var(--panel-2);
    border-radius: 8px;
    font-size: 0.78rem;
    color: var(--muted);
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 5px;
  }
  .route-meta span b { color: var(--text); }
  .model-note {
    margin-top: 8px;
    padding: 9px 10px;
    background: rgba(34, 197, 94, 0.08);
    border: 1px solid rgba(34, 197, 94, 0.3);
    border-radius: 8px;
    font-size: 0.77rem;
    color: #cbd5e1;
  }
  .model-note b { color: #4ade80; }
  .main { display: flex; flex-direction: column; gap: 16px; min-width: 0; }
  .top-row {
    display: grid;
    grid-template-columns: 390px 1fr;
    gap: 16px;
  }
  @media (max-width: 1220px) {
    .top-row { grid-template-columns: 1fr; }
  }
  .kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(165px, 1fr));
    gap: 11px;
  }
  .kpi {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 13px 15px;
  }
  .kpi .lbl { font-size: 0.74rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }
  .kpi .val { font-size: 1.48rem; font-weight: 700; margin-top: 4px; color: var(--text); }
  .kpi .hint { font-size: 0.76rem; color: var(--muted); margin-top: 3px; }
  .kpi.accent .val { color: var(--accent); }
  .kpi.warn .val { color: var(--accent-2); }
  .kpi.green .val { color: var(--green); }
  .kpi.purple .val { color: var(--purple); }
  .chart-header {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
  }
  .chart-header h2 { margin: 0; font-size: 0.98rem; }
  .legend { display: flex; flex-wrap: wrap; gap: 10px; font-size: 0.76rem; color: var(--muted); }
  .legend-item { display: inline-flex; align-items: center; gap: 5px; }
  .swatch { width: 11px; height: 11px; border-radius: 3px; display: inline-block; }
  .chart-wrap { position: relative; width: 100%; overflow-x: auto; }
  svg.chart, svg.map-svg { width: 100%; height: auto; display: block; }
  .map-status {
    font-size: 0.8rem;
    padding: 7px 10px;
    border-radius: 8px;
    background: var(--panel-2);
    color: var(--text);
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }
  .map-btns { display: flex; gap: 6px; }
  button.btn-sm {
    background: var(--panel);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 4px 9px;
    font-size: 0.75rem;
    cursor: pointer;
  }
  button.btn-sm:hover { border-color: var(--accent); color: var(--accent); }
  .tooltip {
    position: absolute;
    pointer-events: none;
    background: rgba(15, 23, 42, 0.96);
    border: 1px solid var(--accent);
    border-radius: 8px;
    padding: 8px 10px;
    font-size: 0.78rem;
    color: var(--text);
    box-shadow: 0 8px 24px rgba(0,0,0,0.45);
    display: none;
    z-index: 20;
    min-width: 185px;
  }
  .table-actions {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
    gap: 10px;
    flex-wrap: wrap;
  }
  button.btn {
    background: var(--accent);
    color: #0b1220;
    border: none;
    border-radius: 8px;
    padding: 8px 14px;
    font-weight: 700;
    font-size: 0.83rem;
    cursor: pointer;
  }
  button.btn:hover { filter: brightness(1.08); }
  .table-scroll { max-height: 340px; overflow: auto; border: 1px solid var(--border); border-radius: 8px; }
  table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
  thead th {
    position: sticky; top: 0;
    background: var(--panel-2);
    color: var(--muted);
    text-align: right;
    padding: 8px 9px;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
  }
  thead th:first-child, thead th:nth-child(2), tbody td:first-child, tbody td:nth-child(2) { text-align: left; }
  tbody td {
    padding: 6px 9px;
    text-align: right;
    border-bottom: 1px solid rgba(51,65,85,0.45);
    white-space: nowrap;
  }
  tbody tr:hover { background: rgba(56, 189, 248, 0.08); }
  .tag-fest {
    background: rgba(245, 158, 11, 0.2);
    color: #fbbf24;
    padding: 2px 5px;
    border-radius: 4px;
    font-size: 0.7rem;
    margin-left: 4px;
  }
  .formula-box {
    font-size: 0.76rem;
    color: var(--muted);
    background: var(--panel-2);
    padding: 9px 11px;
    border-radius: 8px;
    margin-top: 8px;
  }
  .formula-box code { color: var(--accent); }
</style>
</head>
<body>
<header>
  <div>
    <h1>Planeación de Flota y Conductores — Módulo 1 (Demanda Intermunicipal)</h1>
    <div class="sub">Pronóstico a 30 días (marzo 2026) · Comparación transparente entre <b>GLM Poisson Específico (por ruta)</b> y <b>GLM Poisson General (global)</b></div>
  </div>
  <div class="badges">
    <span class="badge esp">Activo: Específico (por ruta)</span>
    <span class="badge" id="badgeTotalRutas">848 rutas · 100% offline</span>
  </div>
</header>

<div class="layout">
  <aside class="sidebar">
    <div class="card">
      <div class="section-title">1. Selección de ruta (texto o mapa)</div>
      <label for="buscarRuta">Buscar municipio, departamento o DIVIPOLA</label>
      <input type="text" id="buscarRuta" placeholder="Ej: Bogotá, Cali, Medellín, Villavicencio..." />
      <label for="selectRuta">Ruta (<span id="contadorRutas">0</span> coincidencias)</label>
      <select id="selectRuta" size="6"></select>
      <div class="route-meta" id="routeMeta"></div>
      <div class="model-note">
        <b>Modelo Específico (por ruta):</b> 1 GLM Poisson entrenado exclusivamente para esta ruta (MAE prueba: 66.55).<br/>
        <b>Modelo General (global):</b> 1 GLM Poisson compartido para las 848 rutas (MAE prueba: 70.86), mostrado como línea gris punteada para comparación.
      </div>
    </div>

    <div class="card">
      <div class="section-title">2. Parámetros operativos</div>

      <label for="cuantilSelect">Cuantil de planeación (riesgo)</label>
      <select id="cuantilSelect">
        <option value="p50">P50 — Pronóstico puntual (mediana)</option>
        <option value="p75">P75 — Riesgo moderado</option>
        <option value="p90" selected>P90 — Alta cobertura (recomendado)</option>
        <option value="p95">P95 — Conservador (pico extremo)</option>
      </select>

      <label for="ocupacionRange">Ocupación objetivo <strong id="ocupacionVal">80%</strong></label>
      <input type="range" id="ocupacionRange" min="50" max="100" step="5" value="80" />

      <label for="claseSelect">Clase de vehículo (capacidad nominal)</label>
      <select id="claseSelect">
        <option value="40" selected>Bus (40 sillas)</option>
        <option value="28">Buseta (28 sillas)</option>
        <option value="16">Microbús (16 sillas)</option>
      </select>

      <label for="capacidadInput">Capacidad sillas (editable)</label>
      <input type="number" id="capacidadInput" min="6" max="80" value="40" />

      <label for="cicloInput">Tiempo de ciclo ida y vuelta (horas)</label>
      <input type="number" id="cicloInput" min="1" max="48" step="0.5" value="8.0" />

      <label for="horasOpRange">Horas de operación al día <strong id="horasOpVal">16 h</strong></label>
      <input type="range" id="horasOpRange" min="8" max="24" step="1" value="16" />

      <label for="jornadaRange">Jornada máxima conductor (CST / Ley 2101) <strong id="jornadaVal">8 h</strong></label>
      <input type="range" id="jornadaRange" min="6" max="12" step="1" value="8" />

      <label for="ausentismoRange">Reserva descansos + ausentismo <strong id="ausentismoVal">8%</strong></label>
      <input type="range" id="ausentismoRange" min="0" max="25" step="1" value="8" />

      <div class="formula-box">
        <div><code>Despachos = ceil(Demanda / (Sillas × Ocup.))</code></div>
        <div><code>Buses = ceil(Despachos / max(0.5, HorasOp / Ciclo))</code></div>
        <div><code>Conductores = ceil(Buses × (HorasOp/Jornada) × (7/6) × (1+Aus.))</code></div>
      </div>
    </div>
  </aside>

  <main class="main">
    <section class="kpi-grid" id="kpiGrid"></section>

    <section class="top-row">
      <div class="card">
        <div class="chart-header">
          <h2>Mapa interactivo de Colombia (clic en 2 puntos)</h2>
        </div>
        <div class="map-status">
          <span id="mapHint">Haz clic en un <b>Origen</b> y luego en un <b>Destino</b>.</span>
          <div class="map-btns">
            <button class="btn-sm" id="btnSwapRoute" title="Invertir origen y destino">Ida ↔ Regreso</button>
            <button class="btn-sm" id="btnClearMap">Limpiar</button>
          </div>
        </div>
        <div class="chart-wrap">
          <svg id="mapColombia" class="map-svg" viewBox="0 0 380 430"></svg>
          <div id="tipMapa" class="tooltip"></div>
        </div>
        <div class="legend" style="margin-top:6px;">
          <span class="legend-item"><span class="swatch" style="background:#22c55e;border-radius:50%"></span>Origen</span>
          <span class="legend-item"><span class="swatch" style="background:#ef4444;border-radius:50%"></span>Destino</span>
          <span class="legend-item"><span class="swatch" style="background:#f59e0b;border-radius:50%"></span>Destinos conectados</span>
        </div>
      </div>

      <div class="card" style="display:flex;flex-direction:column;justify-content:space-between;">
        <div>
          <div class="chart-header">
            <h2 id="tituloDemanda">Pronóstico de demanda a 30 días</h2>
            <div class="legend">
              <span class="legend-item"><span class="swatch" style="background:rgba(56,189,248,0.18);border:1px solid #38bdf8"></span>IC 90% [P05–P95]</span>
              <span class="legend-item"><span class="swatch" style="background:#38bdf8"></span>Específico P50</span>
              <span class="legend-item"><span class="swatch" style="background:#94a3b8"></span>General P50</span>
              <span class="legend-item"><span class="swatch" style="background:#f59e0b"></span>Cuantil activo</span>
              <span class="legend-item"><span class="swatch" style="background:#ef4444"></span>Real</span>
            </div>
          </div>
          <div class="chart-wrap">
            <svg id="chartDemanda" class="chart" viewBox="0 0 820 330"></svg>
            <div id="tipDemanda" class="tooltip"></div>
          </div>
        </div>

        <div style="margin-top:12px;">
          <div class="chart-header">
            <h2>Buses y conductores requeridos por día (30 días)</h2>
            <div class="legend">
              <span class="legend-item"><span class="swatch" style="background:#22c55e"></span>Buses</span>
              <span class="legend-item"><span class="swatch" style="background:#a78bfa"></span>Conductores</span>
            </div>
          </div>
          <div class="chart-wrap">
            <svg id="chartFlota" class="chart" viewBox="0 0 820 210"></svg>
            <div id="tipFlota" class="tooltip"></div>
          </div>
        </div>
      </div>
    </section>

    <section class="card">
      <div class="table-actions">
        <h2 style="margin:0;font-size:0.98rem;">Programación operativa diaria (30 días — Modelo Específico por ruta)</h2>
        <button class="btn" id="btnExportCsv">Descargar tabla CSV</button>
      </div>
      <div class="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Fecha</th>
              <th>Día</th>
              <th>Específico P50</th>
              <th>General P50</th>
              <th id="thCuantil">P90</th>
              <th>IC 90% [P05–P95]</th>
              <th>Real</th>
              <th>Despachos</th>
              <th>Buses</th>
              <th>Conductores</th>
            </tr>
          </thead>
          <tbody id="tablaBody"></tbody>
        </table>
      </div>
    </section>
  </main>
</div>

<script>
const APP_DATA = __APP_DATA_JSON__;

// Indices rapidos por ruta y conexiones origen -> destinos
const routeMap = {};
const adjOut = {};
APP_DATA.routes.forEach(r => {
  routeMap[r.id] = r;
  if (!adjOut[r.oi]) adjOut[r.oi] = [];
  adjOut[r.oi].push(r.di);
});

const mpioMap = {};
APP_DATA.mpios.forEach(m => { mpioMap[m.c] = m; });

const state = {
  rutaId: APP_DATA.routes[0].id,
  filteredRoutes: APP_DATA.routes.slice(),
  mapOrigin: null,   // primer clic en mapa (codigo DIVIPOLA)
  mapDest: null,     // segundo clic en mapa
  lastRows: []
};

const elBuscar = document.getElementById("buscarRuta");
const elSelect = document.getElementById("selectRuta");
const elContador = document.getElementById("contadorRutas");
const elRouteMeta = document.getElementById("routeMeta");
const elMapHint = document.getElementById("mapHint");
const elCuantil = document.getElementById("cuantilSelect");
const elOcup = document.getElementById("ocupacionRange");
const elClase = document.getElementById("claseSelect");
const elCap = document.getElementById("capacidadInput");
const elCiclo = document.getElementById("cicloInput");
const elHorasOp = document.getElementById("horasOpRange");
const elJornada = document.getElementById("jornadaRange");
const elAusent = document.getElementById("ausentismoRange");

document.getElementById("badgeTotalRutas").textContent = `${APP_DATA.routes.length} rutas · 100% offline`;

function fmt(n, dec = 0) {
  return Number(n).toLocaleString("es-CO", {
    minimumFractionDigits: dec,
    maximumFractionDigits: dec
  });
}

function getRouteById(id) {
  return routeMap[id] || APP_DATA.routes[0];
}

function populateRouteList(query) {
  const q = (query || "").trim().toLowerCase();
  state.filteredRoutes = q
    ? APP_DATA.routes.filter(r =>
        r.n.toLowerCase().includes(q) ||
        r.id.toLowerCase().includes(q) ||
        r.o.toLowerCase().includes(q) ||
        r.t.toLowerCase().includes(q))
    : APP_DATA.routes.slice();

  elContador.textContent = state.filteredRoutes.length;
  elSelect.innerHTML = "";
  const limit = Math.min(state.filteredRoutes.length, 250);
  for (let i = 0; i < limit; i++) {
    const r = state.filteredRoutes[i];
    const opt = document.createElement("option");
    opt.value = r.id;
    opt.textContent = `${r.n} (${r.d} km · ~${fmt(r.m)} pax/d)`;
    if (r.id === state.rutaId) opt.selected = true;
    elSelect.appendChild(opt);
  }
}

function selectRoute(id, updateCycle = true) {
  const r = getRouteById(id);
  state.rutaId = r.id;
  state.mapOrigin = r.oi;
  state.mapDest = r.di;
  if (updateCycle) elCiclo.value = r.c;

  elRouteMeta.innerHTML = `
    <span>Código: <b>${r.id}</b></span>
    <span>Distancia: <b>${r.d} km</b></span>
    <span>MAE Específico: <b style="color:#4ade80">${fmt(r.maeE, 1)}</b></span>
    <span>MAE General: <b style="color:#94a3b8">${fmt(r.maeG, 1)}</b></span>
    <span>Cob. IC90: <b>${fmt(r.c90, 1)}%</b></span>
    <span>Regreso: <b>${routeMap[r.di + "_" + r.oi] ? "Disponible" : "No"}</b></span>
  `;
  elMapHint.innerHTML = `Ruta activa: <b style="color:#22c55e">${mpioMap[r.oi].n}</b> → <b style="color:#ef4444">${mpioMap[r.di].n}</b>`;
  renderAll();
}

function handleMapClick(codMpio) {
  const m = mpioMap[codMpio];
  if (!m) return;

  // Si aun no hay origen (o ya habia una ruta completa seleccionada), iniciar nuevo Origen
  if (!state.mapOrigin || (state.mapOrigin && state.mapDest)) {
    state.mapOrigin = codMpio;
    state.mapDest = null;
    const dests = adjOut[codMpio] || [];
    if (dests.length === 0) {
      elMapHint.innerHTML = `<b style="color:#fbbf24">${m.n}</b> solo figura como destino. Elige otro municipio como origen.`;
      state.mapOrigin = null;
    } else {
      elMapHint.innerHTML = `Origen: <b style="color:#22c55e">${m.n}</b> (${dests.length} destinos en amarillo). Haz clic en el <b>Destino</b>.`;
    }
    renderMap();
    return;
  }

  // Segundo clic: seleccionar Destino
  if (codMpio === state.mapOrigin) {
    state.mapOrigin = null;
    state.mapDest = null;
    elMapHint.innerHTML = `Selección limpiada. Haz clic en un <b>Origen</b> y luego en un <b>Destino</b>.`;
    renderMap();
    return;
  }

  const rid = state.mapOrigin + "_" + codMpio;
  if (routeMap[rid]) {
    selectRoute(rid, true);
    populateRouteList(elBuscar.value);
  } else {
    const revId = codMpio + "_" + state.mapOrigin;
    if (routeMap[revId]) {
      selectRoute(revId, true);
      populateRouteList(elBuscar.value);
      elMapHint.innerHTML = `Sin ruta directa en ese sentido; cargada inversa: <b style="color:#22c55e">${mpioMap[codMpio].n}</b> → <b style="color:#ef4444">${mpioMap[state.mapOrigin].n}</b>`;
    } else {
      elMapHint.innerHTML = `No existe ruta directa <b>${mpioMap[state.mapOrigin].n} → ${m.n}</b>. Elige un destino resaltado en <b style="color:#f59e0b">amarillo</b>.`;
    }
  }
}

function renderMap() {
  const svg = document.getElementById("mapColombia");
  const tip = document.getElementById("tipMapa");

  // Proyeccion lineal simple lon [-79.2, -66.8] -> x [14, 366], lat [-4.4, 12.6] -> y [416, 14]
  const lonMin = -79.2, lonMax = -66.8, latMin = -4.4, latMax = 12.6;
  const xOf = lon => 14 + ((lon - lonMin) / (lonMax - lonMin)) * 352;
  const yOf = lat => 416 - ((lat - latMin) / (latMax - latMin)) * 402;

  let html = `<rect x="0" y="0" width="380" height="430" fill="#0b1220" rx="8" />`;

  // Silueta de Colombia
  const polyPts = APP_DATA.border.map(pt => `${xOf(pt[0]).toFixed(1)},${yOf(pt[1]).toFixed(1)}`).join(" ");
  html += `<polygon points="${polyPts}" fill="#17253d" stroke="#475569" stroke-width="1.3" />`;

  // Si hay un origen elegido sin destino, dibujar lineas hacia todos sus destinos conectados
  const connectedSet = new Set(state.mapOrigin ? (adjOut[state.mapOrigin] || []) : []);
  if (state.mapOrigin && !state.mapDest && mpioMap[state.mapOrigin]) {
    const mo = mpioMap[state.mapOrigin];
    connectedSet.forEach(dc => {
      const md = mpioMap[dc];
      if (!md) return;
      html += `<line x1="${xOf(mo.lo)}" y1="${yOf(mo.la)}" x2="${xOf(md.lo)}" y2="${yOf(md.la)}" stroke="rgba(245,158,11,0.35)" stroke-width="1.1" />`;
    });
  }

  // Si hay ruta activa (origen y destino), dibujar flecha destacada
  if (state.mapOrigin && state.mapDest && mpioMap[state.mapOrigin] && mpioMap[state.mapDest]) {
    const mo = mpioMap[state.mapOrigin];
    const md = mpioMap[state.mapDest];
    html += `<line x1="${xOf(mo.lo)}" y1="${yOf(mo.la)}" x2="${xOf(md.lo)}" y2="${yOf(md.la)}" stroke="#38bdf8" stroke-width="2.8" stroke-linecap="round" />`;
  }

  // Dibujar municipios
  APP_DATA.mpios.forEach(m => {
    const cx = xOf(m.lo), cy = yOf(m.la);
    const isOri = m.c === state.mapOrigin;
    const isDst = m.c === state.mapDest;
    const isConn = !state.mapDest && connectedSet.has(m.c);

    let fill = "#64748b", rad = 3.0, stroke = "none";
    if (m.deg >= 12) { fill = "#94a3b8"; rad = 4.2; }
    if (isConn) { fill = "#f59e0b"; rad = 4.8; stroke = "#0f172a"; }
    if (isOri) { fill = "#22c55e"; rad = 6.8; stroke = "#ffffff"; }
    if (isDst) { fill = "#ef4444"; rad = 6.8; stroke = "#ffffff"; }

    html += `<circle data-cod="${m.c}" cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="${rad}" fill="${fill}" stroke="${stroke}" stroke-width="${(isOri || isDst) ? 1.6 : 0.8}" style="cursor:pointer" />`;
    if (isOri || isDst) {
      html += `<text x="${(cx + 8).toFixed(1)}" y="${(cy + 3).toFixed(1)}" fill="#f8fafc" font-size="10" font-weight="700" style="pointer-events:none">${m.n}</text>`;
    }
  });

  svg.innerHTML = html;

  svg.querySelectorAll("circle[data-cod]").forEach(circle => {
    circle.addEventListener("click", () => handleMapClick(circle.dataset.cod));
    circle.addEventListener("mousemove", ev => {
      const m = mpioMap[circle.dataset.cod];
      const nOut = (adjOut[m.c] || []).length;
      tip.style.display = "block";
      tip.innerHTML = `<b>${m.n}</b> (${m.dp})<br>DIVIPOLA: ${m.c}<br>Rutas salientes: <b>${nOut}</b><br><span style="color:#38bdf8">Clic para seleccionar</span>`;
      const bounds = svg.parentElement.getBoundingClientRect();
      tip.style.left = Math.min(bounds.width - 170, Math.max(6, ev.clientX - bounds.left + 10)) + "px";
      tip.style.top = Math.max(6, ev.clientY - bounds.top - 10) + "px";
    });
    circle.addEventListener("mouseleave", () => { tip.style.display = "none"; });
  });
}

function computeSchedule() {
  const qKey = elCuantil.value;
  const qIdx = { p05: 0, p10: 1, p50: 2, p75: 3, p90: 4, p95: 5 }[qKey];
  const ocup = Number(elOcup.value) / 100;
  const cap = Math.max(1, Number(elCap.value) || 40);
  const capEf = Math.max(1, cap * ocup);
  const ciclo = Math.max(0.5, Number(elCiclo.value) || 6);
  const horasOp = Number(elHorasOp.value);
  const jornada = Number(elJornada.value);
  const ausent = Number(elAusent.value) / 100;

  const vueltasDia = Math.max(0.5, horasOp / ciclo);
  const turnosBus = horasOp / jornada;
  const factorReserva = (7 / 6) * (1 + ausent);

  const series = APP_DATA.forecasts[state.rutaId] || [];
  // formato arr: [p05, p10, p50_esp, p75, p90, p95, p50_gen, real]
  return series.map((arr, idx) => {
    const cal = APP_DATA.dates[idx];
    const dem = arr[qIdx];
    const desp = Math.ceil(dem / capEf);
    const buses = Math.ceil(desp / vueltasDia);
    const cond = Math.ceil(buses * turnosBus * factorReserva);
    return {
      fecha: cal[0], dia: cal[1], esFest: cal[2], esPuente: cal[3], festNombre: cal[4],
      p05: arr[0], p10: arr[1], p50: arr[2], p75: arr[3], p90: arr[4], p95: arr[5],
      p50Gen: arr[6], real: arr[7],
      dem, desp, buses, cond, vueltasDia
    };
  });
}

function renderKPIs(rows) {
  const r = getRouteById(state.rutaId);
  const qLabel = elCuantil.value.toUpperCase();
  const totalPax = rows.reduce((a, b) => a + b.dem, 0);
  const avgPax = totalPax / Math.max(1, rows.length);
  const maxRow = rows.reduce((a, b) => (b.dem > a.dem ? b : a), rows[0]);
  const avgDesp = rows.reduce((a, b) => a + b.desp, 0) / rows.length;
  const maxDesp = Math.max(...rows.map(x => x.desp));
  const avgBuses = rows.reduce((a, b) => a + b.buses, 0) / rows.length;
  const maxBuses = Math.max(...rows.map(x => x.buses));
  const avgCond = rows.reduce((a, b) => a + b.cond, 0) / rows.length;
  const maxCond = Math.max(...rows.map(x => x.cond));
  const cob90Mar = rows.filter(x => x.real >= x.p05 && x.real <= x.p95).length / rows.length * 100;

  document.getElementById("kpiGrid").innerHTML = `
    <div class="kpi accent">
      <div class="lbl">Demanda 30 días (${qLabel})</div>
      <div class="val">${fmt(totalPax)}</div>
      <div class="hint">Promedio: ${fmt(avgPax)} pax/día</div>
    </div>
    <div class="kpi warn">
      <div class="lbl">Día pico (${qLabel})</div>
      <div class="val">${fmt(maxRow.dem)}</div>
      <div class="hint">${maxRow.fecha} (${maxRow.dia})</div>
    </div>
    <div class="kpi">
      <div class="lbl">Despachos / día</div>
      <div class="val">${fmt(avgDesp, 1)}</div>
      <div class="hint">Pico: ${fmt(maxDesp)} · ${rows[0].vueltasDia.toFixed(2)} vueltas/bus</div>
    </div>
    <div class="kpi green">
      <div class="lbl">Buses requeridos</div>
      <div class="val">${fmt(avgBuses, 1)} <span style="font-size:0.86rem;color:#94a3b8">/ pico ${maxBuses}</span></div>
      <div class="hint">Extra en pico: +${Math.max(0, maxBuses - Math.round(avgBuses))} buses</div>
    </div>
    <div class="kpi purple">
      <div class="lbl">Conductores requeridos</div>
      <div class="val">${fmt(avgCond, 1)} <span style="font-size:0.86rem;color:#94a3b8">/ pico ${maxCond}</span></div>
      <div class="hint">Con descansos (6×1) y ausentismo</div>
    </div>
    <div class="kpi">
      <div class="lbl">Cobertura IC 90% (mar-26)</div>
      <div class="val">${fmt(cob90Mar, 0)}%</div>
      <div class="hint">MAE Esp: ${fmt(r.maeE, 1)} vs Gen: ${fmt(r.maeG, 1)}</div>
    </div>
  `;
}

function renderDemandaChart(rows) {
  const svg = document.getElementById("chartDemanda");
  const tip = document.getElementById("tipDemanda");
  const r = getRouteById(state.rutaId);
  const qLabel = elCuantil.value.toUpperCase();
  document.getElementById("tituloDemanda").textContent =
    `Pronóstico a 30 días — ${r.n} (Específico por ruta · ${qLabel})`;

  const W = 820, H = 330, padL = 54, padR = 16, padT = 16, padB = 38;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const maxY = Math.max(10, ...rows.map(d => Math.max(d.p95, d.real, d.p50Gen))) * 1.08;
  const xOf = i => padL + (i / Math.max(1, rows.length - 1)) * plotW;
  const yOf = v => padT + plotH - (v / maxY) * plotH;

  let html = "";
  for (let k = 0; k <= 4; k++) {
    const val = (maxY * k) / 4;
    const y = yOf(val);
    html += `<line x1="${padL}" y1="${y}" x2="${W - padR}" y2="${y}" stroke="#334155" stroke-dasharray="3,4" />`;
    html += `<text x="${padL - 7}" y="${y + 4}" fill="#94a3b8" font-size="11" text-anchor="end">${fmt(val)}</text>`;
  }

  const poly90 = [
    ...rows.map((d, i) => `${xOf(i)},${yOf(d.p95)}`),
    ...rows.slice().reverse().map((d, idx) => `${xOf(rows.length - 1 - idx)},${yOf(d.p05)}`)
  ].join(" ");
  html += `<polygon points="${poly90}" fill="rgba(56,189,248,0.16)" />`;

  const poly80 = [
    ...rows.map((d, i) => `${xOf(i)},${yOf(d.p90)}`),
    ...rows.slice().reverse().map((d, idx) => `${xOf(rows.length - 1 - idx)},${yOf(d.p10)}`)
  ].join(" ");
  html += `<polygon points="${poly80}" fill="rgba(56,189,248,0.28)" />`;

  // Modelo General (global) P50 para comparacion visual
  const pathGen = rows.map((d, i) => `${i === 0 ? "M" : "L"}${xOf(i)},${yOf(d.p50Gen)}`).join(" ");
  html += `<path d="${pathGen}" fill="none" stroke="#94a3b8" stroke-width="1.6" stroke-dasharray="4,4" />`;

  // Modelo Especifico (por ruta) P50
  const pathP50 = rows.map((d, i) => `${i === 0 ? "M" : "L"}${xOf(i)},${yOf(d.p50)}`).join(" ");
  html += `<path d="${pathP50}" fill="none" stroke="#38bdf8" stroke-width="2.4" />`;

  // Cuantil activo
  const pathQ = rows.map((d, i) => `${i === 0 ? "M" : "L"}${xOf(i)},${yOf(d.dem)}`).join(" ");
  html += `<path d="${pathQ}" fill="none" stroke="#f59e0b" stroke-width="2.2" stroke-dasharray="6,3" />`;

  // Real observado
  const pathReal = rows.map((d, i) => `${i === 0 ? "M" : "L"}${xOf(i)},${yOf(d.real)}`).join(" ");
  html += `<path d="${pathReal}" fill="none" stroke="#ef4444" stroke-width="1.6" opacity="0.88" />`;

  rows.forEach((d, i) => {
    const cx = xOf(i), cy = yOf(d.real);
    html += `<circle cx="${cx}" cy="${cy}" r="3.1" fill="#ef4444" />`;
    if (i % 5 === 0 || i === rows.length - 1) {
      html += `<text x="${cx}" y="${H - 12}" fill="#94a3b8" font-size="10.5" text-anchor="middle">${d.fecha.slice(5)}</text>`;
    }
    const colW = plotW / rows.length;
    html += `<rect data-idx="${i}" x="${cx - colW / 2}" y="${padT}" width="${colW}" height="${plotH}" fill="transparent" style="cursor:crosshair" />`;
  });

  svg.innerHTML = html;
  svg.querySelectorAll("rect[data-idx]").forEach(rect => {
    rect.addEventListener("mousemove", ev => {
      const d = rows[Number(rect.dataset.idx)];
      tip.style.display = "block";
      tip.innerHTML = `
        <b>${d.fecha} (${d.dia})</b>${d.festNombre ? ` <span class="tag-fest">${d.festNombre}</span>` : ""}<br>
        Real observado: <b style="color:#ef4444">${fmt(d.real)}</b><br>
        Específico (ruta) P50: <b style="color:#38bdf8">${fmt(d.p50)}</b><br>
        General (global) P50: <b style="color:#94a3b8">${fmt(d.p50Gen)}</b><br>
        Cuantil ${qLabel}: <b style="color:#f59e0b">${fmt(d.dem)}</b><br>
        IC 90%: [${fmt(d.p05)} – ${fmt(d.p95)}]<br>
        Buses: <b style="color:#22c55e">${d.buses}</b> · Cond.: <b style="color:#a78bfa">${d.cond}</b>
      `;
      const bounds = svg.parentElement.getBoundingClientRect();
      tip.style.left = Math.min(bounds.width - 210, Math.max(8, ev.clientX - bounds.left + 12)) + "px";
      tip.style.top = "10px";
    });
    rect.addEventListener("mouseleave", () => { tip.style.display = "none"; });
  });
}

function renderFlotaChart(rows) {
  const svg = document.getElementById("chartFlota");
  const tip = document.getElementById("tipFlota");
  const W = 820, H = 210, padL = 48, padR = 16, padT = 14, padB = 34;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const maxY = Math.max(4, ...rows.map(d => d.cond)) * 1.12;
  const slotW = plotW / rows.length;
  const barW = Math.max(4, slotW * 0.38);
  const yOf = v => padT + plotH - (v / maxY) * plotH;

  let html = "";
  for (let k = 0; k <= 4; k++) {
    const val = (maxY * k) / 4;
    const y = yOf(val);
    html += `<line x1="${padL}" y1="${y}" x2="${W - padR}" y2="${y}" stroke="#334155" stroke-dasharray="3,4" />`;
    html += `<text x="${padL - 7}" y="${y + 4}" fill="#94a3b8" font-size="10.5" text-anchor="end">${fmt(val)}</text>`;
  }

  rows.forEach((d, i) => {
    const xBase = padL + i * slotW;
    const hBus = plotH - (yOf(d.buses) - padT);
    const hCond = plotH - (yOf(d.cond) - padT);
    if (d.esPuente || d.esFest) {
      html += `<rect x="${xBase}" y="${padT}" width="${slotW}" height="${plotH}" fill="rgba(245,158,11,0.08)" />`;
    }
    html += `<rect x="${xBase + slotW * 0.1}" y="${yOf(d.buses)}" width="${barW}" height="${hBus}" rx="2" fill="#22c55e" />`;
    html += `<rect x="${xBase + slotW * 0.1 + barW + 1}" y="${yOf(d.cond)}" width="${barW}" height="${hCond}" rx="2" fill="#a78bfa" />`;
    if (i % 5 === 0 || i === rows.length - 1) {
      html += `<text x="${xBase + slotW / 2}" y="${H - 10}" fill="#94a3b8" font-size="10" text-anchor="middle">${d.fecha.slice(5)}</text>`;
    }
    html += `<rect data-idx="${i}" x="${xBase}" y="${padT}" width="${slotW}" height="${plotH}" fill="transparent" style="cursor:pointer" />`;
  });

  svg.innerHTML = html;
  svg.querySelectorAll("rect[data-idx]").forEach(rect => {
    rect.addEventListener("mousemove", ev => {
      const d = rows[Number(rect.dataset.idx)];
      tip.style.display = "block";
      tip.innerHTML = `
        <b>${d.fecha} (${d.dia})</b><br>
        Demanda planeada: <b>${fmt(d.dem)} pax</b><br>
        Despachos: <b>${d.desp}</b><br>
        Buses requeridos: <b style="color:#22c55e">${d.buses}</b><br>
        Conductores requeridos: <b style="color:#a78bfa">${d.cond}</b>
      `;
      const bounds = svg.parentElement.getBoundingClientRect();
      tip.style.left = Math.min(bounds.width - 195, Math.max(8, ev.clientX - bounds.left + 12)) + "px";
      tip.style.top = "8px";
    });
    rect.addEventListener("mouseleave", () => { tip.style.display = "none"; });
  });
}

function renderTable(rows) {
  document.getElementById("thCuantil").textContent = elCuantil.value.toUpperCase();
  const tbody = document.getElementById("tablaBody");
  tbody.innerHTML = rows.map(d => `
    <tr>
      <td>${d.fecha}${d.festNombre ? `<span class="tag-fest">${d.festNombre}</span>` : (d.esPuente ? `<span class="tag-fest">Puente</span>` : "")}</td>
      <td>${d.dia}</td>
      <td style="color:#38bdf8">${fmt(d.p50)}</td>
      <td style="color:#94a3b8">${fmt(d.p50Gen)}</td>
      <td style="color:#f59e0b;font-weight:700">${fmt(d.dem)}</td>
      <td style="color:#94a3b8">[${fmt(d.p05)} – ${fmt(d.p95)}]</td>
      <td style="color:#ef4444">${fmt(d.real)}</td>
      <td>${d.desp}</td>
      <td style="color:#22c55e;font-weight:700">${d.buses}</td>
      <td style="color:#a78bfa;font-weight:700">${d.cond}</td>
    </tr>
  `).join("");
}

function renderAll() {
  document.getElementById("ocupacionVal").textContent = elOcup.value + "%";
  document.getElementById("horasOpVal").textContent = elHorasOp.value + " h";
  document.getElementById("jornadaVal").textContent = elJornada.value + " h";
  document.getElementById("ausentismoVal").textContent = elAusent.value + "%";
  const rows = computeSchedule();
  state.lastRows = rows;
  renderMap();
  renderKPIs(rows);
  renderDemandaChart(rows);
  renderFlotaChart(rows);
  renderTable(rows);
}

elBuscar.addEventListener("input", () => populateRouteList(elBuscar.value));
elSelect.addEventListener("change", () => selectRoute(elSelect.value, true));
elClase.addEventListener("change", () => { elCap.value = elClase.value; renderAll(); });
[elCuantil, elOcup, elCap, elCiclo, elHorasOp, elJornada, elAusent].forEach(el => {
  el.addEventListener("input", renderAll);
  el.addEventListener("change", renderAll);
});

document.getElementById("btnClearMap").addEventListener("click", () => {
  state.mapOrigin = null;
  state.mapDest = null;
  elMapHint.innerHTML = `Haz clic en un <b>Origen</b> y luego en un <b>Destino</b>.`;
  renderMap();
});

document.getElementById("btnSwapRoute").addEventListener("click", () => {
  const r = getRouteById(state.rutaId);
  const revId = r.di + "_" + r.oi;
  if (routeMap[revId]) {
    selectRoute(revId, true);
    populateRouteList(elBuscar.value);
  } else {
    elMapHint.innerHTML = `La ruta inversa <b>${mpioMap[r.di].n} → ${mpioMap[r.oi].n}</b> no está en el catálogo activo.`;
  }
});

document.getElementById("btnExportCsv").addEventListener("click", () => {
  const r = getRouteById(state.rutaId);
  const header = ["ruta_id", "ruta", "fecha", "dia", "p50_especifico", "p50_general", elCuantil.value, "p05", "p95", "real", "despachos", "buses", "conductores"];
  const lines = [header.join(",")].concat(
    state.lastRows.map(d => [r.id, `"${r.n}"`, d.fecha, d.dia, d.p50, d.p50Gen, d.dem, d.p05, d.p95, d.real, d.desp, d.buses, d.cond].join(","))
  );
  const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8;" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `programacion_${r.id}_${elCuantil.value}.csv`;
  a.click();
});

populateRouteList("");
const defaultId = routeMap["11001_76001"] ? "11001_76001" : APP_DATA.routes[0].id;
selectRoute(defaultId, true);
populateRouteList("");
</script>
</body>
</html>
"""


class _TagBalanceChecker(HTMLParser):
    VOID = {"meta", "input", "br", "hr", "img", "link"}

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []
        self.errors: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in self.VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"Tag desbalanceado: cierre </{tag}> con pila {self.stack[-3:]}")
        else:
            self.stack.pop()


def compilar_app_html(out_path: Path | str = OUT_HTML) -> Path:
    """Empaqueta mapa de Colombia + pronosticos de las 848 rutas en un HTML autocontenido."""
    rutas = pd.read_csv(RUTAS_CSV, dtype={"divipola_origen": str, "divipola_destino": str})
    divipola = pd.read_csv(DIVIPOLA_CSV, dtype={"cod_mpio": str})
    cal = pd.read_csv(CAL_CSV)
    pron = pd.read_csv(PRON_CSV)
    met = pd.read_csv(MET_RUTA_CSV)

    rutas = rutas.merge(
        met[["ruta_id", "MAE_general", "MAE_especifico", "Cobertura_IC90_pct"]],
        on="ruta_id",
        how="left",
    )
    fechas_ordenadas = sorted(pron["fecha"].unique().tolist())
    cal_sub = cal[cal["fecha"].isin(fechas_ordenadas)].sort_values("fecha")

    dates_payload = [
        [
            str(r["fecha"]),
            str(r["dia_semana"])[:3],
            int(r["es_festivo"]),
            int(r["es_puente"]),
            str(r["festivo_nombre"]) if pd.notna(r["festivo_nombre"]) else "",
        ]
        for _, r in cal_sub.iterrows()
    ]

    forecasts_payload: dict[str, list[list[float]]] = {}
    for rid, grp in pron.sort_values(["ruta_id", "fecha"]).groupby("ruta_id"):
        forecasts_payload[str(rid)] = [
            [
                round(float(row.p05)),
                round(float(row.p10)),
                round(float(row.p50)),
                round(float(row.p75)),
                round(float(row.p90)),
                round(float(row.p95)),
                round(float(row.p50_general)),
                int(row.real),
            ]
            for row in grp.itertuples(index=False)
        ]

    routes_payload = []
    mpio_deg: dict[str, int] = {}
    for row in rutas.itertuples(index=False):
        if str(row.ruta_id) not in forecasts_payload:
            continue
        oi = str(row.divipola_origen).zfill(5)
        di = str(row.divipola_destino).zfill(5)
        mpio_deg[oi] = mpio_deg.get(oi, 0) + 1
        mpio_deg[di] = mpio_deg.get(di, 0) + 1
        ciclo = estimar_tiempo_ciclo_horas(float(row.dist_linea_recta_km))
        routes_payload.append({
            "id": str(row.ruta_id),
            "oi": oi,
            "di": di,
            "n": str(row.ruta_nombre),
            "o": str(row.dpto_origen),
            "t": str(row.dpto_destino),
            "d": round(float(row.dist_linea_recta_km), 1),
            "c": round(ciclo, 1),
            "m": round(float(row.media_pax_origen_dia), 1),
            "maeE": round(float(row.MAE_especifico), 1) if pd.notna(row.MAE_especifico) else 0.0,
            "maeG": round(float(row.MAE_general), 1) if pd.notna(row.MAE_general) else 0.0,
            "c90": round(float(row.Cobertura_IC90_pct), 1) if pd.notna(row.Cobertura_IC90_pct) else 0.0,
        })

    # Municipios unicos con coordenadas para el mapa interactivo de Colombia
    div_idx = divipola.set_index("cod_mpio")
    mpios_payload = []
    for cod, deg in sorted(mpio_deg.items(), key=lambda x: x[1]):
        if cod not in div_idx.index:
            continue
        mr = div_idx.loc[cod]
        if pd.isna(mr["latitud"]) or pd.isna(mr["longitud"]):
            continue
        mpios_payload.append({
            "c": cod,
            "n": str(mr["nom_mpio"]),
            "dp": str(mr["dpto"]),
            "la": round(float(mr["latitud"]), 4),
            "lo": round(float(mr["longitud"]), 4),
            "deg": deg,
        })

    app_data = {
        "modelName": str(pron["modelo"].iloc[0]),
        "border": COLOMBIA_BORDER_LONLAT,
        "mpios": mpios_payload,
        "dates": dates_payload,
        "routes": routes_payload,
        "forecasts": forecasts_payload,
    }
    json_str = json.dumps(app_data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html_final = HTML_TEMPLATE.replace("__APP_DATA_JSON__", json_str)

    checker = _TagBalanceChecker()
    checker.feed(html_final)
    if checker.errors or checker.stack:
        raise RuntimeError(f"Errores de compilacion HTML: {checker.errors} | pila pendiente: {checker.stack}")
    json.loads(json_str.replace("<\\/", "</"))

    path = Path(out_path)
    path.write_text(html_final, encoding="utf-8")
    return path


if __name__ == "__main__":
    p = compilar_app_html()
    print(f"HTML compilado y verificado: {p} ({p.stat().st_size / 1024:.1f} KB)")
