"""Generate PDF report of NHL SOG and MLB Pitcher Strikeout model analysis."""
from fpdf import FPDF

class Report(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 5, "Model Feature Importance & Walk-Forward Analysis", align="C")
            self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(20, 60, 120)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(20, 60, 120)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def sub_title(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(40, 40, 40)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, txt):
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5, txt)
        self.ln(2)

    def add_table(self, headers, data, col_widths=None):
        if col_widths is None:
            col_widths = [(self.w - self.l_margin - self.r_margin) / len(headers)] * len(headers)
        # Header row
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(20, 60, 120)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True, align="C")
        self.ln()
        # Data rows
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(30, 30, 30)
        for row_idx, row in enumerate(data):
            if row_idx % 2 == 0:
                self.set_fill_color(240, 245, 255)
            else:
                self.set_fill_color(255, 255, 255)
            for i, val in enumerate(row):
                align = "L" if i == 0 or (isinstance(val, str) and not val.replace('.','').replace('-','').replace('+','').replace('%','').replace('$','').isdigit()) else "C"
                self.cell(col_widths[i], 6, str(val), border=1, fill=True, align=align)
            self.ln()
        self.ln(3)


def build_report():
    pdf = Report()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ── TITLE PAGE ──
    pdf.add_page()
    pdf.ln(50)
    pdf.set_font("Helvetica", "B", 26)
    pdf.set_text_color(20, 60, 120)
    pdf.cell(0, 15, "Model Analysis Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 10, "Feature Importance & Walk-Forward Meta-Analysis", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_draw_color(20, 60, 120)
    pdf.line(60, pdf.get_y(), pdf.w - 60, pdf.get_y())
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, "NHL Shots on Goal (SOG) Predictor", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "MLB Pitcher Strikeout Predictor", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(20)
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 8, "Generated March 2026", align="C", new_x="LMARGIN", new_y="NEXT")

    # ══════════════════════════════════════════════════════════════
    #  NHL SOG MODEL
    # ══════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(20, 60, 120)
    pdf.cell(0, 12, "NHL Shots on Goal Model", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # ── Features ──
    pdf.section_title("Feature Importance (18 Features)")
    pdf.body_text(
        "The model uses separate XGBoost models for forwards and defensemen. "
        "Target variable is the residual (actual_sog - baseline_sog). "
        "Baseline formula: 0.55 x season + 0.30 x last10 + 0.15 x last5."
    )

    nhl_features = [
        ["baseline_sog",         "0.1560", "0.1509"],
        ["pct_games_3plus",      "0.0893", "0.0919"],
        ["toi_last_5",           "0.0715", "0.0642"],
        ["player_cv",            "0.0701", "0.0700"],
        ["is_back_to_back",      "0.0651", "0.0954"],
        ["avg_toi",              "0.0634", "0.0579"],
        ["implied_team_total",   "0.0570", "0.0600"],
        ["avg_shift_length",     "0.0564", "0.0000"],
        ["game_total",           "0.0556", "0.0546"],
        ["arena_bias",           "0.0543", "0.0000"],
        ["opp_shots_allowed_pos","  --",   "0.0542"],
        ["rolling_pp_rate",      "0.0000", "0.0538"],
        ["sharp_consensus_prob", "0.0000", "0.0000"],
        ["sog_prop_line",        "0.0000", "0.0000"],
        ["opp_shots_allowed",    "0.0000", "0.0000"],
        ["is_home",              "0.0000", "0.0000"],
        ["rest_days",            "0.0000", "0.0000"],
        ["linemate_quality",     "0.0000", "0.0000"],
    ]
    pdf.add_table(
        ["Feature", "Fwd Importance", "Def Importance"],
        nhl_features,
        col_widths=[75, 50, 50],
    )

    pdf.body_text(
        "Key insight: baseline_sog dominates both models (~15%). "
        "Back-to-back impact is notably larger for defensemen (9.5% vs 6.5%). "
        "Arena bias matters for forwards but not defense; rolling PP rate matters for defense but not forwards."
    )

    # ── Model Performance ──
    pdf.section_title("Model Performance")
    perf = [
        ["Forward Model",  "25,370 / 2,633", "1.041", "1.316", "76.3%", "88.5%"],
        ["Defense Model",  "12,825 / 1,327", "0.887", "1.133", "85.1%", "94.7%"],
        ["Combined",       "38,195 / 3,960", "0.989", "--",    "--",    "--"],
    ]
    pdf.add_table(
        ["Model", "Train/Test", "MAE", "RMSE", "Acc >2.5", "Acc >3.5"],
        perf,
        col_widths=[30, 35, 25, 25, 30, 30],
    )

    # ── Walk-Forward ──
    pdf.section_title("Walk-Forward Meta-Analysis: Optimal Strategy")
    pdf.sub_title("Strategy: BMG Blended Unders")
    pdf.body_text(
        "Blended probability (50% model + 50% sharp book consensus) targeting UNDERS only "
        "on soft sportsbooks (BetMGM / PlayAlberta). Quarter Kelly sizing (0.25 fraction). "
        "Expanding training window with 2-week rolling test periods. "
        "102 strategy variants evaluated across 7 categories."
    )
    wf_nhl = [
        ["Edge Threshold",    "5%",      "3%"],
        ["Yield",             "+14.3%",  "+6.7%"],
        ["Bet Side",          "Unders",  "Unders"],
        ["Target Book",       "BetMGM / PlayAlberta", "BetMGM / PlayAlberta"],
        ["Kelly Fraction",    "0.25",    "0.25"],
    ]
    pdf.add_table(
        ["Metric", "Aggressive", "Inclusive"],
        wf_nhl,
        col_widths=[55, 55, 55],
    )
    pdf.body_text(
        "Sharp books used for consensus: BetOnlineAg, DraftKings, FanDuel. "
        "Soft books targeted: BetMGM, William Hill (US). "
        "PlayAlberta vig adjustment averages ~0.42pp difference from BetMGM."
    )

    # ══════════════════════════════════════════════════════════════
    #  MLB PITCHER STRIKEOUT MODEL
    # ══════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(20, 60, 120)
    pdf.cell(0, 12, "MLB Pitcher Strikeout Model", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    pdf.body_text(
        "Architecture: Dual XGBoost (BF x K/BF) with Monte Carlo distribution layer. "
        "Final prediction = predicted_BF x predicted_K_per_BF. "
        "Training data: 4,062 samples (Mar 29 - Sep 16, 2024). Test: 327 samples."
    )

    # ── BF Model Features ──
    pdf.section_title("BF Model Features (21 Features) - Top 10")
    bf_features = [
        ["baseline_bf",      "0.1139"],
        ["days_rest",        "0.0852"],
        ["avg_pitch_count",  "0.0633"],
        ["market_k_line",    "0.0518"],
        ["pitches_last",     "0.0507"],
        ["is_home",          "0.0478"],
        ["rolling_3_pc",     "0.0465"],
        ["innings_last",     "0.0457"],
        ["rolling_whip",     "0.0455"],
        ["pitches_per_bf",   "0.0449"],
    ]
    pdf.add_table(
        ["Feature", "Importance"],
        bf_features,
        col_widths=[80, 50],
    )
    pdf.body_text(
        "Other BF features: bf_trend, season_bb_rate, park_k_factor, rolling_velocity, "
        "pitches_per_ip, rolling_walk_rate, opp_contact_rate, implied_team_win_prob, "
        "game_total_line, team_moneyline, sharp_consensus_prob."
    )

    # ── KBF Model Features ──
    pdf.section_title("KBF Model Features (38 Features) - Top 12")
    kbf_features = [
        ["baseline_k_rate",           "0.0858"],
        ["market_k_line",             "0.0601"],
        ["k_minus_bb_rate",           "0.0589"],
        ["two_strike_putaway_rate",   "0.0561"],
        ["opp_k_rate",                "0.0518"],
        ["opp_contact_rate",          "0.0361"],
        ["pitcher_cv",                "0.0327"],
        ["is_home",                   "0.0310"],
        ["ff_usage",                  "0.0245"],
        ["csw_rate",                  "0.0237"],
        ["days_rest",                 "0.0231"],
        ["matchup_k_rate",            "0.0225"],
    ]
    pdf.add_table(
        ["Feature", "Importance"],
        kbf_features,
        col_widths=[80, 50],
    )
    pdf.body_text(
        "Other KBF features: whiff_rate, zone_rate, chase_rate, first_pitch_strike_rate, "
        "zone_contact_rate, pitch-type usage & whiff rates (ff/sl/ch/cu), velo_gap, "
        "pitch_entropy, breaking_usage, matchup features (whiff/k/contact rates), "
        "lineup-specific whiff interactions, tto_k_decay, rolling_velocity, sharp_consensus_prob."
    )

    # ── Model Performance ──
    pdf.section_title("Model Performance")
    mlb_perf = [
        ["MAE",                      "1.681"],
        ["RMSE",                     "2.158"],
        ["Mean Bias",                "-0.047"],
        ["Market MAE",               "1.698"],
        ["Model vs Market Win Rate", "52.9%"],
        ["Model vs Market MAE Edge", "0.017"],
    ]
    pdf.add_table(
        ["Metric", "Value"],
        mlb_perf,
        col_widths=[80, 50],
    )

    pdf.sub_title("Threshold Accuracy")
    thresh = [
        ["Over 3.5", "74.9%"],
        ["Over 4.5", "67.0%"],
        ["Over 5.5", "66.1%"],
        ["Over 6.5", "76.5%"],
        ["Over 7.5", "86.5%"],
    ]
    pdf.add_table(["Threshold", "Accuracy"], thresh, col_widths=[60, 40])

    pdf.sub_title("Performance by Pitcher Tier")
    tiers = [
        ["Low K (< 4)",   "64",  "1.473", "+0.119"],
        ["Mid K (4-6)",   "209", "1.699", "-0.185"],
        ["High K (6-8)",  "54",  "1.858", "+0.290"],
    ]
    pdf.add_table(["Tier", "n", "MAE", "Bias"], tiers, col_widths=[45, 25, 30, 30])

    # ── Walk-Forward ──
    pdf.section_title("Walk-Forward Meta-Analysis: Optimal Strategy")
    pdf.sub_title("Strategy: Unders 3.5 Only")
    wf_mlb_opt = [
        ["Number of Bets",    "124"],
        ["Win Rate",          "57.3%"],
        ["Yield",             "+17.9%"],
        ["ROI",               "+107.6%  ($100 -> $207.56)"],
        ["Max Drawdown",      "37.8%"],
        ["Peak Bankroll",     "$315.42"],
        ["Trough",            "$92.57"],
    ]
    pdf.add_table(["Metric", "Value"], wf_mlb_opt, col_widths=[55, 80])

    pdf.sub_title("Monthly Breakdown")
    monthly = [
        ["May 2024",  "7",  "4",  "57.1%", "+$1.99",   "+5.6%"],
        ["Jun 2024",  "67", "41", "61.2%", "+$110.21",  "+24.9%"],
        ["Jul 2024",  "44", "24", "54.5%", "+$2.78",    "+0.6%"],
        ["Aug 2024",  "6",  "2",  "33.3%", "-$7.42",    "-7.3%"],
    ]
    pdf.add_table(
        ["Month", "Bets", "Wins", "Win %", "P&L", "Yield"],
        monthly,
        col_widths=[28, 18, 18, 22, 28, 22],
    )

    pdf.sub_title("Runner-Up: Unders 3.5 + 4.5, Any +EV")
    runner = [
        ["Number of Bets", "425"],
        ["Win Rate",       "54.8%"],
        ["Yield",          "+9.1%"],
        ["ROI",            "+208.1%  ($100 -> $308.13)"],
        ["Max Drawdown",   "62.4%"],
    ]
    pdf.add_table(["Metric", "Value"], runner, col_widths=[55, 80])

    pdf.sub_title("Key Findings")
    pdf.body_text(
        "- Edge is concentrated in UNDERS on low lines (3.5, 4.5) only.\n"
        "- Overs at any line: -11.9% yield (money-losing).\n"
        "- Unders at 5.5+: also money-losing (-11.0% to -15.4% yield).\n"
        "- All +EV bets unfiltered: -1.2% yield -- strategy filtering is essential.\n"
        "- June was by far the strongest month; August consistently negative."
    )

    # ── Save ──
    path = r"C:\Users\rober\OneDrive\Documents\GitHub\nhl-sog-predictor\Model_Analysis_Report.pdf"
    pdf.output(path)
    return path


if __name__ == "__main__":
    p = build_report()
    print(f"PDF saved to: {p}")
