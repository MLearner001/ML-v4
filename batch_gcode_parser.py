"""batch_gcode_parser.py

Tahap 1: Parsing File NC (.mpf / .nc) dan Ekstraksi Fitur Kinematika 3D.
(Updated: V2 dengan Physics-Informed MAX_RAPID Limit)
"""

from dataclasses import dataclass, field
import math
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class MachineState:
  # Koordinat Translasi Aktual (mm)
  x: float = 0.0
  y: float = 0.0
  z: float = 0.0

  # Sudut Sumbu Rotasi Aktual (Derajat)
  b: float = 0.0
  c: float = 0.0

  # Orientasi Vektor Pahat 5-Axis (TRAORI)
  a3: float = 0.0
  b3: float = 0.0
  c3: float = 1.0

  # Commanded Values
  cmd_f: float = 1000.0
  cmd_s: float = 0.0

  # Modal Motions
  motion_mode: str = "G01"  # G00, G01, G02, G03
  is_absolute: bool = True  # G90 (Absolute) = True, G91 (Incremental) = False
  is_traori: int = 0

  # Modal Cycles
  c832_tol: float = 0.1
  c832_mode: int = 0
  c800_rotx: float = 0.0
  c800_roty: float = 0.0
  c800_rotz: float = 0.0

  # R-Parameters Storage
  r_params: Dict[int, float] = field(default_factory=dict)

  # MCALL Drilling State
  is_mcall_active: bool = False
  mcall_params: Dict[str, float] = field(default_factory=dict)

  # Kinematic History untuk Vektor & Reversal
  prev_dx: float = 0.0
  prev_dy: float = 0.0
  prev_dz: float = 0.0


class NCParser:

  def __init__(self):
    self.state = MachineState()
    self.MAX_RAPID = 20000.0  # [V2 UPDATE] Limit aktual untuk G00

  def _evaluate_r_param(self, val_str: str) -> float:
    """Mengevaluasi nilai numerik atau variabel R-parameter (misal: R1, R10)."""
    val_str = val_str.strip().upper()
    if val_str.startswith("R"):
      try:
        r_idx = int(val_str[1:])
        return self.state.r_params.get(r_idx, 0.0)
      except ValueError:
        pass
    try:
      return float(val_str)
    except ValueError:
      return 0.0

  def _parse_r_assignments(self, line: str):
    """Mendeteksi penugasan parameter R (contoh: R1=1500, R2=0.05)."""
    matches = re.findall(r"R(\d+)\s*=\s*([-+]?\d*\.?\d+)", line, re.IGNORECASE)
    for idx, val in matches:
      self.state.r_params[int(idx)] = float(val)

  def _parse_cycle832(self, line: str):
    """Mengekstrak toleransi dan mode dari CYCLE832(0.005, 3, 1) atau makro string."""
    match = re.search(r"CYCLE832\s*\(([^)]+)\)", line, re.IGNORECASE)
    if match:
      args = [a.strip() for a in match.group(1).split(",")]
      if len(args) >= 1 and args[0]:
        try:
          self.state.c832_tol = float(args[0])
        except ValueError:
          pass

      if len(args) >= 2 and args[1]:
        mode_str = args[1].upper()
        try:
          self.state.c832_mode = int(float(mode_str))
        except ValueError:
          # Mapping makro Siemens umum ke integer
          if "_OFF" in mode_str:
            self.state.c832_mode = 0
          elif "_ROUGH" in mode_str:
            self.state.c832_mode = 1
          elif "_SEMIFIN" in mode_str:
            self.state.c832_mode = 2
          elif "_FINISH" in mode_str:
            self.state.c832_mode = 3
          else:
            self.state.c832_mode = 0

  def _parse_cycle800(self, line: str):
    """Mengekstrak parameter rotasi orientasi bidang CYCLE800."""
    match = re.search(r"CYCLE800\s*\(([^)]+)\)", line, re.IGNORECASE)
    if match:
      args = [a.strip() for a in match.group(1).split(",")]
      # Siemens: Parameter rotasi sudut bidang berada di argumen ke-4, 5, 6 atau variasi template
      # Default fallbacks jika argumen berurutan:
      try:
        if len(args) >= 7:
          self.state.c800_rotx = float(args[4]) if args[4] else 0.0
          self.state.c800_roty = float(args[5]) if args[5] else 0.0
          self.state.c800_rotz = float(args[6]) if args[6] else 0.0
      except ValueError:
        pass

  def _parse_mcall(self, line: str):
    """Mengekstrak definisi siklus modal bor MCALL CYCLE81/82/83."""
    clean_line = line.strip().upper()
    if "MCALL" in clean_line:
      if "CYCLE" in clean_line:
        match = re.search(r"CYCLE\d+\s*\(([^)]*)\)", clean_line)
        if match:
          raw_args = match.group(1).split(",")
          args = [
              float(arg.strip()) if arg.strip() else None
              for arg in raw_args
          ]
          args.extend([None] * (6 - len(args)))

          rtp = args[0] if args[0] is not None else 0.0
          rfp = args[1] if args[1] is not None else 0.0
          sdis = args[2] if args[2] is not None else 0.0
          dp = args[3]
          dpr = args[4]

          if dp is None and dpr is not None:
             dp_val = rfp - abs(dpr)
          elif dp is not None:
             dp_val = dp
          else:
             dp_val = 0.0

          self.state.mcall_params = {
              "rtp": rtp,
              "rfp": rfp,
              "sdis": sdis,
              "dp": dp_val,
          }
          self.state.is_mcall_active = True
      else:
        # MCALL tanpa argumen = Membatalkan MCALL
        self.state.is_mcall_active = False
        self.state.mcall_params = {}

  def _calculate_sharpness_angle(
      self, v1: Tuple[float, float, float], v2: Tuple[float, float, float]
  ) -> float:
    """Menghitung sudut transisi lintasan antar dua blok berurutan (Radian)."""
    norm1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2)
    norm2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2 + v2[2] ** 2)

    if norm1 < 1e-6 or norm2 < 1e-6:
      return 0.0

    dot_prod = (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]) / (
        norm1 * norm2
    )
    dot_prod = max(-1.0, min(1.0, dot_prod))
    return math.acos(dot_prod)

  def parse_file(self, filepath: str, out_dir: Optional[str] = None) -> pd.DataFrame:
    """Membaca file .mpf/.nc dan mengembalikan DataFrame fitur baris per baris."""
    import os
    parsed_rows = []

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
      lines = f.readlines()

    # Absolute sequential execution index to avoid N99999 reset issues
    absolute_idx = 1

    for raw_line in lines:
      line = raw_line.strip()
      # Hapus komentar (ditandai dengan semicolon ;)
      if ";" in line:
        line = line.split(";")[0].strip()
      if not line:
        continue

      # 1. Ekstraksi Block ID (Nomor Baris N)
      block_match = re.search(r"^N(\d+)", line, re.IGNORECASE)
      n_number = int(block_match.group(1)) if block_match else None

      block_id = str(absolute_idx)

      # Kita simpan N number untuk referensi, tapi index absolut yang jadi Block_ID

      # 2. Parsing Modal & R-Assignments
      self._parse_r_assignments(line)
      self._parse_cycle832(line)
      self._parse_cycle800(line)
      self._parse_mcall(line)

      # 3. Cek Perubahan Mode Modal
      if re.search(r"\bTRAORI\b", line, re.IGNORECASE):
        self.state.is_traori = 1
      if re.search(r"\bTRAFOOF\b", line, re.IGNORECASE):
        self.state.is_traori = 0

      if re.search(r"\bG0*0\b", line, re.IGNORECASE):
        self.state.motion_mode = "G00"
      elif re.search(r"\bG0*1\b", line, re.IGNORECASE):
        self.state.motion_mode = "G01"
      elif re.search(r"\bG0*2\b", line, re.IGNORECASE):
        self.state.motion_mode = "G02"
      elif re.search(r"\bG0*3\b", line, re.IGNORECASE):
        self.state.motion_mode = "G03"

      if re.search(r"\bG90\b", line, re.IGNORECASE):
        self.state.is_absolute = True
      elif re.search(r"\bG91\b", line, re.IGNORECASE):
        self.state.is_absolute = False

      # 4. Commanded Feedrate & Spindle
      # G04 F... signifies Dwell Time, NOT modal feedrate.
      if not re.search(r"\bG0*4\b", line, re.IGNORECASE):
        f_match = re.search(r"\bF\s*=\s*([^\s,]+)|\bF([0-9\.]+)", line)
        if f_match:
          f_val_str = f_match.group(1) if f_match.group(1) else f_match.group(2)
          self.state.cmd_f = self._evaluate_r_param(f_val_str)

      s_match = re.search(r"\bS\s*=\s*([^\s,]+)|\bS([0-9\.]+)", line)
      if s_match:
        s_val_str = s_match.group(1) if s_match.group(1) else s_match.group(2)
        self.state.cmd_s = self._evaluate_r_param(s_val_str)

      # 5. Ekstraksi Koordinat Target
      x_match = re.search(r"\bX\s*=\s*([^\s,]+)|\bX([-\d\.]+)", line)
      y_match = re.search(r"\bY\s*=\s*([^\s,]+)|\bY([-\d\.]+)", line)
      z_match = re.search(r"\bZ\s*=\s*([^\s,]+)|\bZ([-\d\.]+)", line)
      b_match = re.search(r"\bB\s*=\s*([^\s,]+)|\bB([-\d\.]+)", line)
      c_match = re.search(r"\bC\s*=\s*([^\s,]+)|\bC([-\d\.]+)", line)
      a3_match = re.search(r"\bA3\s*=\s*([-\d\.]+)", line)
      b3_match = re.search(r"\bB3\s*=\s*([-\d\.]+)", line)
      c3_match = re.search(r"\bC3\s*=\s*([-\d\.]+)", line)

      # 6. Ekstraksi Parameter Melingkar (I, J, K, CR)
      i_match = re.search(r"\bI\s*=\s*AC\(([-\d\.]+)\)|\bI([-\d\.]+)", line)
      j_match = re.search(r"\bJ\s*=\s*AC\(([-\d\.]+)\)|\bJ([-\d\.]+)", line)
      k_match = re.search(r"\bK\s*=\s*AC\(([-\d\.]+)\)|\bK([-\d\.]+)", line)
      cr_match = re.search(r"\bCR\s*=\s*([-\d\.]+)", line)

      def get_target(match, current_val):
          if match:
              val = self._evaluate_r_param(match.group(1) if match.group(1) else match.group(2))
              return val if self.state.is_absolute else current_val + val
          return current_val

      tgt_x = get_target(x_match, self.state.x)
      tgt_y = get_target(y_match, self.state.y)
      tgt_z = get_target(z_match, self.state.z)
      tgt_b = get_target(b_match, self.state.b)
      tgt_c = get_target(c_match, self.state.c)

      tgt_a3 = float(a3_match.group(1)) if a3_match else self.state.a3
      tgt_b3 = float(b3_match.group(1)) if b3_match else self.state.b3
      tgt_c3 = float(c3_match.group(1)) if c3_match else self.state.c3

      # -------------------------------------------------------------
      # KASUS A: EKSPANSI GEOMETRIS MCALL (DRILLING)
      # -------------------------------------------------------------
      if self.state.is_mcall_active and (x_match or y_match):
        rtp = self.state.mcall_params["rtp"]
        rfp = self.state.mcall_params["rfp"]
        sdis = self.state.mcall_params["sdis"]
        dp = self.state.mcall_params["dp"]
        z_approach = rfp + sdis

        # 4 Sub-Gerakan Pengeboran
        sub_movements = [
            # 1. Gerak Rapid XY di Return Plane
            (
                tgt_x,
                tgt_y,
                rtp,
                "G00",
                self.state.cmd_f,
                f"{block_id}_pos",
            ),
            # 2. Gerak Rapid Turun Z ke Approach Plane
            (
                tgt_x,
                tgt_y,
                z_approach,
                "G00",
                self.state.cmd_f,
                f"{block_id}_app",
            ),
            # 3. Gerak Pemakanan Bor Z ke Kedalaman Akhir DP
            (tgt_x, tgt_y, dp, "G01", self.state.cmd_f, f"{block_id}_cut"),
            # 4. Gerak Retract Cepat Naik Z kembali ke RTP
            (
                tgt_x,
                tgt_y,
                rtp,
                "G00",
                self.state.cmd_f,
                f"{block_id}_ret",
            ),
        ]

        for sub_x, sub_y, sub_z, mode, f_val, sub_id in sub_movements:
          dx = sub_x - self.state.x
          dy = sub_y - self.state.y
          dz = sub_z - self.state.z
          delta_3d = math.sqrt(dx**2 + dy**2 + dz**2)
          angle = self._calculate_sharpness_angle(
              (self.state.prev_dx, self.state.prev_dy, self.state.prev_dz),
              (dx, dy, dz),
          )

          # [V2 UPDATE] Set MAX_RAPID untuk G00 di dalam sub-gerakan
          effective_f = self.MAX_RAPID if mode == "G00" else f_val

          # [Fase 2] Kinematic Blend Ratio (Rotasi per Translasi)
          blend_ratio = 0.0 / (delta_3d + 1e-6)

          # [Fase 2 Update] Theoretical Duration (Seconds)
          # Asumsi minimal velocity 1.0 mm/min untuk menghindari div by zero
          safe_f = max(1.0, effective_f)
          theo_duration = (delta_3d / safe_f) * 60.0

          parsed_rows.append({
              "Block_ID": sub_id,
              "N_Number": n_number if n_number else -1,
              "Cmd_F": effective_f,
              "Cmd_S": self.state.cmd_s,
              "Is_G01": 1 if mode == "G01" else 0,
              "Is_G02": 0,
              "Is_G03": 0,
              "Is_Traori": self.state.is_traori,
              "Is_Cycle800": (
                  1
                  if (
                      self.state.c800_rotx
                      or self.state.c800_roty
                      or self.state.c800_rotz
                  )
                  else 0
              ),
              "Is_MCALL_Sub": 1,
              "C832_Tol": self.state.c832_tol,
              "C832_Mode": self.state.c832_mode,
              "Tgt_X": sub_x,
              "Tgt_Y": sub_y,
              "Tgt_Z": sub_z,
              "Tgt_B": tgt_b,
              "Tgt_C": tgt_c,
              "Delta_3D": delta_3d,
              "Delta_Rot": 0.0,
              "Tool_Vector_Delta": 0.0,
              "Kinematic_Blend_Ratio": blend_ratio,
              "Sharpness_Angle": angle,
              "Is_Motion_Block": 1 if delta_3d > 1e-4 else 0,
              "Is_Reversal_X": (
                  1 if (dx * self.state.prev_dx < 0 and abs(dx) > 1e-4) else 0
              ),
              "Is_Reversal_Y": (
                  1 if (dy * self.state.prev_dy < 0 and abs(dy) > 1e-4) else 0
              ),
              "Is_Reversal_Z": (
                  1 if (dz * self.state.prev_dz < 0 and abs(dz) > 1e-4) else 0
              ),
              "Theo_Duration": theo_duration,
          })

          # Update state
          self.state.x, self.state.y, self.state.z = sub_x, sub_y, sub_z
          self.state.prev_dx, self.state.prev_dy, self.state.prev_dz = (
              dx,
              dy,
              dz,
          )

        absolute_idx += 1
        continue

      # -------------------------------------------------------------
      # KASUS B: GERAKAN KONTUR & INDEXING STANDAR
      # -------------------------------------------------------------
      dx = tgt_x - self.state.x
      dy = tgt_y - self.state.y
      dz = tgt_z - self.state.z
      db = tgt_b - self.state.b
      dc = tgt_c - self.state.c
      da3 = tgt_a3 - self.state.a3
      db3 = tgt_b3 - self.state.b3
      dc3 = tgt_c3 - self.state.c3

      delta_3d = math.sqrt(dx**2 + dy**2 + dz**2)
      delta_rot = math.sqrt(db**2 + dc**2)
      tool_vec_delta = math.sqrt(da3**2 + db3**2 + dc3**2)

      # Circular motions (G02/G03) are smooth arcs, not sharp lines.
      if self.state.motion_mode in ["G02", "G03"]:
        sharpness_angle = 0.0
        is_reversal_x = 0
        is_reversal_y = 0
        is_reversal_z = 0

        # Calculate true arc length
        radius = 0.0
        if cr_match:
            radius = abs(float(cr_match.group(1)))
        elif i_match or j_match or k_match:
            # I, J, K usually represent offset from start point to center
            i_val = float(i_match.group(1) if i_match.group(1) else i_match.group(2)) if i_match else 0.0
            j_val = float(j_match.group(1) if j_match.group(1) else j_match.group(2)) if j_match else 0.0
            k_val = float(k_match.group(1) if k_match.group(1) else k_match.group(2)) if k_match else 0.0

            # Check if AC() absolute coordinates were used
            if i_match and i_match.group(1):
                i_val = i_val - self.state.x
            if j_match and j_match.group(1):
                j_val = j_val - self.state.y
            if k_match and k_match.group(1):
                k_val = k_val - self.state.z

            radius = math.sqrt(i_val**2 + j_val**2 + k_val**2)

        if radius > 0 and delta_3d > 0:
            # Full circle edge case (start == end, delta_3d == 0) is handled separately below if needed.
            # However, typically a full circle has delta_3d = 0, but if it's a full circle, G-code often provides I/J/K without X/Y/Z.
            # If delta_3d == 0 and radius > 0, it's a full circle: arc_length = 2 * pi * R
            # Wait, if start == end, delta_3d is 0.
            if delta_3d < 1e-4:
                delta_3d = 2 * math.pi * radius
            else:
                # s = R * theta
                # chord = 2 * R * sin(theta/2) -> theta = 2 * arcsin(chord / (2R))
                ratio = delta_3d / (2.0 * radius)
                ratio = max(-1.0, min(1.0, ratio)) # Clip to avoid domain errors
                theta = 2.0 * math.asin(ratio)
                delta_3d = radius * theta
        elif radius > 0 and delta_3d < 1e-4:
            # Full circle
            delta_3d = 2 * math.pi * radius

      else:
        sharpness_angle = self._calculate_sharpness_angle(
            (self.state.prev_dx, self.state.prev_dy, self.state.prev_dz),
            (dx, dy, dz),
        )

        is_reversal_x = (
            1 if (dx * self.state.prev_dx < 0 and abs(dx) > 1e-4) else 0
        )
        is_reversal_y = (
            1 if (dy * self.state.prev_dy < 0 and abs(dy) > 1e-4) else 0
        )
        is_reversal_z = (
            1 if (dz * self.state.prev_dz < 0 and abs(dz) > 1e-4) else 0
        )

      is_motion = (
          1 if (delta_3d > 1e-4 or delta_rot > 1e-4 or tool_vec_delta > 1e-4) else 0
      )

      # [V2 UPDATE] Set limit untuk G00 vs G01
      effective_limit_f = self.MAX_RAPID if self.state.motion_mode == "G00" else self.state.cmd_f

      # [Fase 2] Kinematic Blend Ratio
      blend_ratio = delta_rot / (delta_3d + 1e-6)

      # [Fase 2 Update] Theoretical Duration (Seconds)
      safe_f = max(1.0, effective_limit_f)
      # Jika hanya rotasi tanpa translasi, kita menggunakan estimasi kecepatan rotasi B/C maksimal (misal 5000 derajat/menit)
      if delta_3d < 1e-4 and delta_rot > 1e-4:
          theo_duration = (delta_rot / 5000.0) * 60.0
      else:
          theo_duration = (delta_3d / safe_f) * 60.0

      parsed_rows.append({
          "Block_ID": str(block_id),
          "N_Number": n_number if n_number else -1,
          "Cmd_F": effective_limit_f,
          "Cmd_S": self.state.cmd_s,
          "Is_G01": 1 if self.state.motion_mode == "G01" else 0,
          "Is_G02": 1 if self.state.motion_mode == "G02" else 0,
          "Is_G03": 1 if self.state.motion_mode == "G03" else 0,
          "Is_Traori": self.state.is_traori,
          "Is_Cycle800": (
              1
              if (
                  self.state.c800_rotx
                  or self.state.c800_roty
                  or self.state.c800_rotz
              )
              else 0
          ),
          "Is_MCALL_Sub": 0,
          "C832_Tol": self.state.c832_tol,
          "C832_Mode": self.state.c832_mode,
          "Tgt_X": tgt_x,
          "Tgt_Y": tgt_y,
          "Tgt_Z": tgt_z,
          "Tgt_B": tgt_b,
          "Tgt_C": tgt_c,
          "Delta_3D": delta_3d,
          "Delta_Rot": delta_rot,
          "Tool_Vector_Delta": tool_vec_delta,
          "Kinematic_Blend_Ratio": blend_ratio,
          "Sharpness_Angle": sharpness_angle,
          "Is_Motion_Block": is_motion,
          "Is_Reversal_X": is_reversal_x,
          "Is_Reversal_Y": is_reversal_y,
          "Is_Reversal_Z": is_reversal_z,
          "Theo_Duration": theo_duration,
      })

      # Update State
      self.state.x, self.state.y, self.state.z = tgt_x, tgt_y, tgt_z
      self.state.b, self.state.c = tgt_b, tgt_c
      self.state.a3, self.state.b3, self.state.c3 = tgt_a3, tgt_b3, tgt_c3
      if delta_3d > 1e-4:
        self.state.prev_dx, self.state.prev_dy, self.state.prev_dz = (
            dx,
            dy,
            dz,
        )

      absolute_idx += 1

    return pd.DataFrame(parsed_rows)


if __name__ == "__main__":
  import sys
  import os

  parser = NCParser()
  # Contoh penggunaan parsing
  input_file = (
      sys.argv[1] if len(sys.argv) > 1 else "sample_5axis_part.mpf"
  )
  try:
    df_result = parser.parse_file(input_file)

    base_name = os.path.basename(input_file)
    output_csv = base_name.replace(".mpf", "_parsed.csv").replace(
        ".nc", "_parsed.csv"
    )
    df_result.to_csv(output_csv, index=False)
    print(
        f"[SUCCESS] Berhasil mengekstrak {len(df_result)} baris fitur ke"
        f" {output_csv}"
    )
  except FileNotFoundError:
    print(
        f"[INFO] File {input_file} tidak ditemukan. Silakan jalankan: python"
        " batch_gcode_parser.py <path_file.mpf>"
    )
