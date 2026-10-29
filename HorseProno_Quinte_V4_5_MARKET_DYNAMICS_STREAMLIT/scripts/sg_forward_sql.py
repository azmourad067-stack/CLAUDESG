"""Génère les requêtes SQL (lecture seule) d'évaluation prospective dans Supabase.

Pourquoi en SQL : les cotes horodatées pré-course (forward_model_predictions,
snapshot T-30 à T-10) et les rapports définitifs PMU (quinte_payouts_raw)
sont stockés dans Supabase. Ces requêtes recalculent dans la base, à partir
de l'artefact JSON gelé, exactement les mêmes variables que features.py et
musique.py, puis produisent :

- runners    : variables et utilité fondamentale par partant (test de parité) ;
- grid       : log-vraisemblance de l'étape 2 sur une grille (α, γ) ;
- payout     : rapport définitif / cote snapshot des gagnants, par tranche ;
- candidates : meilleur candidat EV par course (+ favori) pour α, γ, facteur ;
- logloss    : log-loss marché vs modèle par discipline.

Usage :
    python scripts/sg_forward_sql.py --artifact models/x.json --mode grid \
        --start 2026-09-12 --end 2026-09-30 > grid.sql

Le code n'écrit rien dans la base.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simple_gagnant import musique as M  # noqa: E402
from simple_gagnant.features import FAMILY_FEATURES  # noqa: E402

FAMILY_SQL_LETTERS = {fam: sorted(letters) for fam, letters in M.FAMILY_LETTERS.items()}


def _num(v: float) -> str:
    return repr(float(v))


def utility_sql(artifact: dict) -> str:
    """Utilité fondamentale sous forme linéaire compacte : Σ (b/s)·x − Σ b·μ/s.

    Les coefficients sont écrits avec 12 chiffres significatifs (écart < 1e-9
    par rapport au calcul Python, vérifié par le test de parité).
    """
    cases = []
    for fam, m in artifact["stage1"].items():
        const = 0.0
        terms = []
        for f, mu, sc, b in zip(m["features"], m["mean"], m["scale"], m["coef"]):
            if abs(b) < 1e-12:
                continue
            w = b / sc
            const -= w * mu
            terms.append(f"{w:.12g}*f.{f}")
        expr = " + ".join(terms) if terms else "0"
        cases.append(f"when '{fam}' then {const:.12g} + {expr}")
    return "case f.family " + " ".join(cases) + " else 0 end"


def base_cte(artifact: dict, start: str, end: str) -> str:
    fam_case = (
        "case r.discipline when 'PLAT' then 'PLAT' "
        "when 'ATTELE_AUTOSTART' then 'TROT' when 'ATTELE_VOLTE' then 'TROT' when 'TROT_MONTE' then 'TROT' "
        "when 'HAIES' then 'OBSTACLE' when 'STEEPLECHASE' then 'OBSTACLE' when 'CROSS_COUNTRY' then 'OBSTACLE' "
        "else 'AUTRE' end"
    )
    letters = {fam: "array[" + ",".join(f"'{x}'" for x in ls) + "]" for fam, ls in FAMILY_SQL_LETTERS.items()}
    w, ps, pw, pp, pi = (
        _num(M.PRIOR_STRENGTH), _num(M.PRIOR_SCORE), _num(M.PRIOR_WIN), _num(M.PRIOR_PLACE), _num(M.PRIOR_INCIDENT)
    )
    return f"""
with sg_raw as materialized (
  select q.race_date, q.meeting_number, q.race_number, b.value->>'rembourse' as refunded,
         r.value->'combinaison'->>0 as comb, (r.value->>'dividendePourUnEuro')::numeric/100.0 as dividend
  from quinte_payouts_raw q
  cross join lateral jsonb_array_elements(q.payload) b
  cross join lateral jsonb_array_elements(b.value->'rapports') r
  where b.value->>'typePari' = 'E_SIMPLE_GAGNANT'
    and q.race_date between date '{start}' and date '{end}'
),
sg_race as materialized (
  select race_date, meeting_number, race_number,
         count(*) filter (where comb ~ '^\\d+$') as n_winners,
         max(case when comb ~ '^\\d+$' then comb::int end) as winner,
         max(case when comb ~ '^\\d+$' then dividend end) as dividend,
         bool_or(coalesce(refunded,'false') = 'true') as refunded
  from sg_raw group by 1,2,3
),
snap as materialized (
  select distinct on (participant_id) participant_id, race_id, odds_snapshot, predicted_at, scheduled_start
  from forward_model_predictions
  order by participant_id, model_version_id
),
race_snap as materialized (
  select race_id, min(predicted_at) as predicted_at, min(scheduled_start) as scheduled_start
  from snap group by race_id
),
runners as materialized (
  select p.id as pid, r.id as race_id, r.race_date, r.meeting_number, r.race_number, r.hippodrome,
         r.discipline, {fam_case} as family, r.distance_m,
         p.horse_number, p.weight_kg, p.draw, p.age, upper(coalesce(p.sex,'')) as sex,
         coalesce(p.recent_form,'') as recent_form,
         s.odds_snapshot as odds,
         extract(epoch from (rs.scheduled_start - rs.predicted_at))/60.0 as minutes_before,
         sr.winner, sr.dividend
  from races r
  join race_snap rs on rs.race_id = r.id
  join participants p on p.race_id = r.id and not coalesce(p.is_non_runner,false)
  left join snap s on s.participant_id = p.id
  join sg_race sr on sr.race_date = r.race_date and sr.meeting_number = r.meeting_number
                 and sr.race_number = r.race_number and sr.n_winners = 1 and not sr.refunded
  where r.race_date between date '{start}' and date '{end}'
),
race_stats as materialized (
  select race_id, count(*) as n_runners, count(odds) as n_odds,
         sum(case when odds > 1 then 1.0/odds end) as booksum,
         percentile_cont(0.5) within group (order by weight_kg) filter (where weight_kg between 30 and 90) as w_med,
         avg(age) filter (where age between 2 and 20) as age_mean,
         count(*) filter (where horse_number = winner) as winner_present
  from runners group by race_id
),
tok as (
  select ru.pid, t.ord, t.m[1] as res, lower(t.m[2]) as dl,
         sum(case when t.m[1] is null then 1 else 0 end) over (partition by ru.pid order by t.ord) as season
  from runners ru
  cross join lateral regexp_matches(regexp_replace(ru.recent_form, '\\s', '', 'g'),
       '\\(\\d{{2,4}}\\)|(\\d{{1,2}}|[A-Za-z])([A-Za-z])', 'g') with ordinality as t(m, ord)
),
tok2 as (
  select pid, season, dl,
         row_number() over (partition by pid order by ord) as tk,
         case when res ~ '^\\d+$' then res::int end as pos
  from tok where res is not null
),
tok3 as materialized (
  select pid, season, dl, tk, pos,
         case when pos between 1 and 9 then (10 - pos)/9.0 else 0 end as score
  from tok2
),
mus as materialized (
  select ru.pid,
         count(t.tk) as n_all,
         count(t.tk) filter (where t.season = 0) as n_cur,
         count(t.tk) filter (where t.tk <= {M.WINDOW}) as k,
         sum(t.score) filter (where t.tk <= {M.WINDOW}) as s_score,
         max(t.score) filter (where t.tk = 1) as last_score,
         count(t.tk) filter (where t.tk <= {M.WINDOW} and t.pos = 1) as n_win,
         count(t.tk) filter (where t.tk <= {M.WINDOW} and t.pos between 1 and 3) as n_place,
         count(t.tk) filter (where t.tk <= {M.WINDOW} and t.pos is null) as n_inc,
         count(t.tk) filter (where t.tk <= {M.WINDOW} and (
              (ru.family = 'PLAT' and t.dl = any({letters['PLAT']})) or
              (ru.family = 'TROT' and t.dl = any({letters['TROT']})) or
              (ru.family = 'OBSTACLE' and t.dl = any({letters['OBSTACLE']})))) as n_same
  from runners ru left join tok3 t on t.pid = ru.pid
  group by ru.pid
),
f as materialized (
  select ru.*, rs.n_runners, rs.n_odds, rs.booksum, rs.winner_present,
    least(m.n_all, 10)/10.0 as mus_n,
    least(m.n_cur, 10)/10.0 as mus_cur,
    case when m.k = 0 then 1.0 else 0.0 end as mus_missing,
    case when m.k = 0 then {ps} else (m.s_score + {w}*{ps})/(m.k + {w}) end as mus_score5,
    case when m.k = 0 then {ps} else m.last_score end as mus_last,
    case when m.k = 0 then {pw} else (m.n_win + {w}*{pw})/(m.k + {w}) end as mus_win5,
    case when m.k = 0 then {pp} else (m.n_place + {w}*{pp})/(m.k + {w}) end as mus_place5,
    case when m.k = 0 then {pi} else (m.n_inc + {w}*{pi})/(m.k + {w}) end as mus_inc5,
    case when m.k = 0 then 0.0 else m.n_same::numeric/m.k end as mus_samefam5,
    coalesce(case when ru.weight_kg between 30 and 90 then (ru.weight_kg - rs.w_med)/5.0 end, 0.0) as weight_rel,
    case when ru.weight_kg between 30 and 90 then 0.0 else 1.0 end as weight_missing,
    case when ru.discipline in ('PLAT','ATTELE_AUTOSTART') and ru.draw >= 1
         then least(greatest((ru.draw - 1.0)/greatest(rs.n_runners - 1.0, 1.0), 0.0), 1.5) else 0.0 end as draw_rel,
    coalesce(least(greatest(case when ru.age between 2 and 20 then ru.age - rs.age_mean end, -6.0), 6.0), 0.0) as age_rel,
    case when ru.age between 2 and 20 then 0.0 else 1.0 end as age_missing,
    case when ru.sex like 'F%' then 1.0 else 0.0 end as sex_female,
    case when ru.sex like 'H%' then 1.0 else 0.0 end as sex_gelding
  from runners ru join race_stats rs on rs.race_id = ru.race_id join mus m on m.pid = ru.pid
),
scored as (
  select f.*, ({utility_sql(artifact)}) as u_fund,
         case when f.n_odds = f.n_runners and f.odds > 1 then ln((1.0/f.odds)/f.booksum) end as lpm,
         (f.horse_number = f.winner) as won
  from f
),
complete as materialized (
  select * from scored
  where n_odds = n_runners and n_runners >= 4 and winner_present = 1 and family <> 'AUTRE'
)"""


def payout_case(edges: list[float], factors: list[float], col: str = "odds") -> str:
    parts = []
    for lo, hi, fac in zip(edges[:-1], edges[1:], factors):
        parts.append(f"when {col} >= {_num(lo)} and {col} < {_num(hi)} then {_num(fac)}")
    return "case " + " ".join(parts) + f" else {_num(factors[-1])} end"


def q_runners(base: str, limit_races: int) -> str:
    return base + f"""
select race_id, race_date, meeting_number, race_number, discipline, horse_number, weight_kg, draw, age, sex,
       recent_form, odds, round(u_fund::numeric, 6) u_fund, round(mus_score5::numeric,6) mus_score5,
       round(mus_inc5::numeric,6) mus_inc5, round(weight_rel::numeric,6) weight_rel,
       round(draw_rel::numeric,6) draw_rel, round(age_rel::numeric,6) age_rel
from complete where race_id in (select distinct race_id from complete order by race_id limit {limit_races})
order by race_id, horse_number;"""


def q_grid(base: str, alphas: list[float], gammas: list[float]) -> str:
    a_vals = ",".join(f"({_num(a)})" for a in alphas)
    g_vals = ",".join(f"({_num(g)})" for g in gammas)
    return base + f"""
, grid as (select a.v as alpha, g.v as gamma from (values {a_vals}) a(v) cross join (values {g_vals}) g(v)),
lik as (
  select gr.alpha, gr.gamma, c.race_id,
         sum(case when c.won then gr.alpha*c.lpm + gr.gamma*c.u_fund else 0 end)
         - ln(sum(exp(gr.alpha*c.lpm + gr.gamma*c.u_fund))) as ll
  from complete c cross join grid gr group by 1,2,3
)
select string_agg(alpha||':'||gamma||':'||round(ll::numeric,3), ' ' order by alpha, gamma) grid, max(races) races
from (select alpha, gamma, count(*) races, sum(ll) ll from lik group by 1,2) z;"""


def q_payout(base: str, edges: list[float]) -> str:
    buckets = " ".join(
        f"when odds >= {_num(lo)} and odds < {_num(hi)} then {i}" for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:]))
    )
    return base + f"""
, w as (select odds, dividend, minutes_before, ln(dividend/odds) lr, case {buckets} end as bucket from complete where won)
select bucket, count(*) n, round(avg(lr)::numeric,4) mean_lr,
       round((percentile_cont(0.25) within group (order by lr))::numeric,4) q25_lr,
       round((percentile_cont(0.5) within group (order by lr))::numeric,4) med_lr,
       round(stddev(lr)::numeric,4) sd_lr, round(avg(odds)::numeric,2) mean_odds,
       round(avg(minutes_before)::numeric,1) mb
from w group by rollup(bucket) order by bucket;"""


def q_candidates(base: str, alpha: float, gamma: float, corners: list[tuple[float, float]],
                 edges: list[float], factors: list[float], low_factors: list[float], ev_floor: float) -> str:
    corner_cols = []
    for i, (a, g) in enumerate(corners):
        corner_cols.append(
            f"exp({_num(a)}*lpm + {_num(g)}*u_fund) / sum(exp({_num(a)}*lpm + {_num(g)}*u_fund)) over (partition by race_id) as pc{i}"
        )
    least_expr = "least(" + ",".join(f"pc{i}" for i in range(len(corners))) + ")" if corners else "p"
    return base + f"""
, prob as (
  select c.*,
         exp({_num(alpha)}*lpm + {_num(gamma)}*u_fund) / sum(exp({_num(alpha)}*lpm + {_num(gamma)}*u_fund)) over (partition by race_id) as p,
         exp(lpm) as pm,
         {", ".join(corner_cols) if corner_cols else "1 as pc0"},
         {payout_case(edges, factors)} as k,
         {payout_case(edges, low_factors)} as k_low
  from complete c
),
ev as (
  select prob.*, {least_expr} as p_low,
         p*odds*k - 1 as ev, {least_expr}*odds*k - 1 as ev_low, p*odds*k_low - 1 as ev_drift,
         row_number() over (partition by race_id order by p*odds*k desc, horse_number) as ev_rank,
         row_number() over (partition by race_id order by odds asc, horse_number) as fav_rank,
         row_number() over (partition by race_id order by p desc, horse_number) as model_rank
  from prob
),
race_level as (
  select race_id,
         sum(case when won then ln(p) end) as ll_model, sum(case when won then lpm end) as ll_market,
         -sum(p*ln(p)) as entropy, max(abs(p - pm)) as max_disagreement
  from ev group by race_id
),
fav as (select race_id, horse_number fav_num, odds fav_odds, won fav_won, dividend fav_div from ev where fav_rank = 1)
select e.race_date, e.meeting_number m, e.race_number c, e.discipline d, e.n_runners n,
       round(e.booksum::numeric,3) bk, round(e.minutes_before::numeric,1) mb,
       e.horse_number h, e.odds o, round(e.pm::numeric,4) pm, round(e.p::numeric,4) p, round(e.p_low::numeric,4) pl,
       round(e.ev::numeric,4) ev, round(e.ev_low::numeric,4) evl, round(e.ev_drift::numeric,4) evd,
       e.ev_rank er, e.model_rank mr, case when e.won then 1 else 0 end w, case when e.won then e.dividend end div,
       round(rl.ll_model::numeric,4) llm, round(rl.ll_market::numeric,4) llk,
       round(rl.entropy::numeric,3) ent, round(rl.max_disagreement::numeric,3) dis,
       fv.fav_num fh, fv.fav_odds fo, case when fv.fav_won then 1 else 0 end fw, fv.fav_div fd
from ev e join race_level rl using (race_id) join fav fv using (race_id)
where e.ev_rank = 1 or e.ev >= {_num(ev_floor)}
order by e.race_date, e.meeting_number, e.race_number, e.ev_rank;"""


def ev_ctes(alpha: float, gamma: float, corners: list[tuple[float, float]], edges: list[float],
            factors: list[float], low_factors: list[float], sel: dict) -> str:
    """CTE communes : probabilités, scénarios de paramètres, EV, portes de qualité (= selector.py)."""
    corner_cols = ",\n         ".join(
        f"exp({_num(a)}*lpm + {_num(g)}*u_fund) / sum(exp({_num(a)}*lpm + {_num(g)}*u_fund)) over (partition by race_id) as pc{i}"
        for i, (a, g) in enumerate(corners)
    )
    # Identique à SimpleGagnantModel.predict : minimum sur le point central et les 4 coins.
    least_expr = "least(p," + ",".join(f"pc{i}" for i in range(len(corners))) + ")"
    return f"""
, prob as (
  select c.*,
         exp({_num(alpha)}*lpm + {_num(gamma)}*u_fund) / sum(exp({_num(alpha)}*lpm + {_num(gamma)}*u_fund)) over (partition by race_id) as p,
         exp(lpm) as pm,
         {corner_cols},
         {payout_case(edges, factors)} as k,
         {payout_case(edges, low_factors)} as k_low
  from complete c
),
ev as materialized (
  select prob.*, {least_expr} as p_low,
         p*odds*k - 1 as ev, {least_expr}*odds*k - 1 as ev_low, p*odds*k_low - 1 as ev_drift,
         row_number() over (partition by race_id order by odds asc, horse_number) as fav_rank,
         row_number() over (partition by race_id order by p desc, horse_number) as model_rank
  from prob
),
gate as materialized (
  select race_id, max(n_runners) n, max(booksum) bk, avg(mus_missing) miss
  from ev group by race_id
),
ok as materialized (
  select race_id from gate
  where n between {sel['min_runners']} and {sel['max_runners']}
    and bk between {_num(sel['booksum_min'])} and {_num(sel['booksum_max'])}
    and miss <= {_num(sel['max_missing_musique_share'])}
)"""


def q_policy(base: str, ctes: str, p_mins: list[float], odds_maxs: list[float], ev_ts: list[float]) -> str:
    pv = ",".join(f"({_num(x)})" for x in p_mins)
    ov = ",".join(f"({_num(x)})" for x in odds_maxs)
    tv = ",".join(f"({_num(x)})" for x in ev_ts)
    return base + ctes + f"""
, pc as (select * from (values {pv}) v(pmin)),
oc as (select * from (values {ov}) v(omax)),
cand as (
  select pc.pmin, oc.omax, e.race_id, e.family, e.odds, e.p, e.ev, e.ev_low, e.ev_drift, e.won, e.dividend,
         row_number() over (partition by pc.pmin, oc.omax, e.race_id order by e.ev desc, e.p desc, e.horse_number) rk
  from ev e join ok using (race_id) cross join pc cross join oc
  where e.p >= pc.pmin and e.odds <= oc.omax
),
th as (select * from (values {tv}) v(evt)),
rb as (select * from (values (0),(1),(2)) v(robust)),
bets as (
  select b.pmin, b.omax, th.evt, rb.robust, b.won, case when b.won then b.dividend - 1 else -1 end as pnl
  from cand b cross join th cross join rb
  where b.rk = 1 and b.ev >= th.evt and b.ev > 0
    and (rb.robust = 0 or (rb.robust = 1 and b.ev_low >= 0) or (rb.robust = 2 and b.ev_low >= 0 and b.ev_drift >= 0))
),
agg as (
  select pmin, omax, evt, robust, count(*) n, sum(case when won then 1 else 0 end) wins,
         sum(pnl) profit, coalesce(stddev(pnl), 0) sd
  from bets group by 1,2,3,4
)
select (select count(*) from ok) races_ok, (select count(*) from gate) races_all,
       string_agg(concat_ws(':', pmin, omax, evt, robust, n, wins, round(profit::numeric,2), round(sd::numeric,3)), ' '
                  order by pmin, omax, evt, robust) policies
from agg;"""


def q_test(base: str, ctes: str, sel: dict) -> str:
    """Évaluation de la période de test avec la configuration gelée (sorties compactes)."""
    return base + ctes + f"""
, cand as (
  select e.*, row_number() over (partition by e.race_id order by e.ev desc, e.p desc, e.horse_number) rk
  from ev e join ok using (race_id)
  where e.p >= {_num(sel['p_min'])} and e.odds <= {_num(sel['odds_max'])}
),
best as (select * from cand where rk = 1),
status as (
  select g.race_id,
    case when o.race_id is null then 'NO_BET_GATE'
         when b.race_id is null then 'NO_BET_NONE'
         when b.ev <= {_num(sel['ev_watch'])} or b.ev <= 0 then 'NO_BET_EV'
         when b.ev < {_num(sel['ev_test'])} then 'WATCH'
         when b.ev >= {_num(sel['ev_validated'])} and b.ev_low >= 0 and b.ev_drift >= 0 then 'ROBUST'
         else 'TEST' end as st
  from gate g left join ok o using (race_id) left join best b using (race_id)
),
bets as (
  select b.*, s.st from best b join status s using (race_id) where s.st in ('TEST','ROBUST')
),
fav as (
  select e.race_id, e.won, e.dividend from ev e join ok using (race_id) where e.fav_rank = 1
),
top as (
  select e.race_id, e.won, e.dividend from ev e join ok using (race_id) where e.model_rank = 1
)
select
  (select string_agg(st||':'||n, ' ') from (select st, count(*) n from status group by st) z) as status_counts,
  (select concat_ws(':', count(*), sum(case when won then 1 else 0 end), round(sum(case when won then dividend-1 else -1 end)::numeric,2)) from fav) as favourite,
  (select concat_ws(':', count(*), sum(case when won then 1 else 0 end), round(sum(case when won then dividend-1 else -1 end)::numeric,2)) from top) as model_top1,
  (select string_agg(concat_ws('|', race_date, meeting_number, race_number, discipline, n_runners, horse_number, odds,
          round(p::numeric,4), round(pm::numeric,4), round(ev::numeric,4), round(ev_low::numeric,4), round(ev_drift::numeric,4),
          case when won then 1 else 0 end, coalesce(dividend::text,''), st, round(minutes_before::numeric,1)), ';'
          order by race_date, meeting_number, race_number) from bets) as bets;"""


def q_logloss(base: str, alpha: float, gamma: float) -> str:
    return base + f"""
, prob as (
  select c.*, exp({_num(alpha)}*lpm + {_num(gamma)}*u_fund) / sum(exp({_num(alpha)}*lpm + {_num(gamma)}*u_fund)) over (partition by race_id) as p,
         exp(u_fund) / sum(exp(u_fund)) over (partition by race_id) as pf
  from complete c
)
select coalesce(discipline,'TOUTES') discipline, count(distinct race_id) races,
       round((-sum(case when won then lpm end)/count(distinct race_id))::numeric,4) ll_market,
       round((-sum(case when won then ln(p) end)/count(distinct race_id))::numeric,4) ll_model,
       round((-sum(case when won then ln(pf) end)/count(distinct race_id))::numeric,4) ll_fund,
       round((-sum(case when won then ln(1.0/n_runners) end)/count(distinct race_id))::numeric,4) ll_uniform
from prob group by rollup(discipline) order by races desc;"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--mode", required=True,
                    choices=["runners", "grid", "payout", "candidates", "logloss", "policy", "test"])
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--alphas", default="0.9,1.0,1.1,1.2,1.3")
    ap.add_argument("--gammas", default="-0.3,-0.15,0,0.15,0.3")
    ap.add_argument("--limit-races", type=int, default=5)
    ap.add_argument("--ev-floor", type=float, default=0.0)
    ap.add_argument("--selector", default=None, help="configuration JSON du sélecteur (modes policy/test)")
    ap.add_argument("--p-mins", default="0,0.05,0.1,0.15")
    ap.add_argument("--odds-maxs", default="8,15,30,1000")
    ap.add_argument("--ev-thresholds", default="0,0.05,0.1,0.15,0.2,0.3")
    args = ap.parse_args()

    artifact = json.loads(Path(args.artifact).read_text(encoding="utf-8"))
    base = base_cte(artifact, args.start, args.end)
    comb = artifact.get("combination", {})
    payout = artifact.get("payout") or {}
    edges = payout.get("edges", [1.0, 1e9])
    factors = payout.get("factors", [1.0] * (len(edges) - 1))
    lows = payout.get("low_factors", factors)
    if args.mode in ("policy", "test"):
        from simple_gagnant.config import SelectorConfig
        from simple_gagnant.model import MarketCombination

        sel = (SelectorConfig.load(args.selector) if args.selector else SelectorConfig()).to_dict()
        mc = MarketCombination.from_dict(comb)
        ctes = ev_ctes(mc.alpha, mc.gamma, mc.parameter_corners()[1:], edges, factors, lows, sel)
        if args.mode == "policy":
            sql = q_policy(base, ctes, [float(x) for x in args.p_mins.split(",")],
                           [float(x) for x in args.odds_maxs.split(",")],
                           [float(x) for x in args.ev_thresholds.split(",")])
        else:
            sql = q_test(base, ctes, sel)
        sys.stdout.write(sql.strip() + "\n")
        return
    if args.mode == "runners":
        sql = q_runners(base, args.limit_races)
    elif args.mode == "grid":
        sql = q_grid(base, [float(x) for x in args.alphas.split(",")], [float(x) for x in args.gammas.split(",")])
    elif args.mode == "payout":
        sql = q_payout(base, edges)
    elif args.mode == "candidates":
        from simple_gagnant.model import MarketCombination

        mc = MarketCombination.from_dict(comb)
        sql = q_candidates(base, mc.alpha, mc.gamma, mc.parameter_corners()[1:], edges, factors, lows, args.ev_floor)
    else:
        mc_a = float(comb.get("alpha", 1.0))
        mc_g = float(comb.get("gamma", 0.0))
        sql = q_logloss(base, mc_a, mc_g)
    sys.stdout.write(sql.strip() + "\n")


if __name__ == "__main__":
    main()
