-- HorseProno Simple Gagnant — journal d'audit des prédictions (migration additive).
-- N'altère ni ne supprime aucune table existante. À appliquer une seule fois.
--
-- Principes :
--   * sg_predictions / sg_prediction_runners : photographie pré-course, immuable
--     (UPDATE et DELETE bloqués par trigger, y compris pour le rôle service) ;
--   * sg_settlements : règlement au rapport définitif PMU, enregistrement séparé ;
--   * lecture publique (comme les autres tables HorseProno), écriture réservée
--     au rôle service (clé serveur), qui contourne RLS.

create table if not exists public.sg_predictions (
  snapshot_id        text primary key,
  captured_at        timestamptz not null,
  race_date          date not null,
  meeting_number     integer not null check (meeting_number > 0),
  race_number        integer not null check (race_number > 0),
  race_id            text not null,
  hippodrome         text,
  discipline         text,
  family             text,
  scheduled_start    timestamptz,
  minutes_to_start   numeric,
  runners            integer,
  status             text not null check (status in (
                       'NO BET — COURSE ÉCARTÉE', 'SURVEILLANCE', 'CANDIDAT À TESTER',
                       'OPPORTUNITÉ VALIDÉE PAR LE MODÈLE')),
  reasons            text,
  reason_text        text,
  candidate_number   integer,
  candidate_name     text,
  candidate_odds     numeric check (candidate_odds is null or candidate_odds > 1),
  p_model            numeric check (p_model is null or (p_model >= 0 and p_model <= 1)),
  p_low              numeric,
  p_market           numeric,
  fair_odds          numeric,
  break_even_odds    numeric,
  payout_factor      numeric,
  ev                 numeric,
  ev_low             numeric,
  ev_drift           numeric,
  edge               numeric,
  booksum            numeric,
  entropy_norm       numeric,
  kl_model_market    numeric,
  arguments_for      jsonb,
  arguments_against  jsonb,
  alerts             jsonb,
  non_runners        jsonb,
  model_fingerprint  text not null,
  model_version      text,
  selector_json      jsonb,
  engine_version     text,
  mode               text not null default 'OBSERVATION',
  inserted_at        timestamptz not null default now()
);

create index if not exists idx_sg_predictions_race on public.sg_predictions (race_date, meeting_number, race_number);
create index if not exists idx_sg_predictions_status on public.sg_predictions (status);

create table if not exists public.sg_prediction_runners (
  snapshot_id       text not null references public.sg_predictions(snapshot_id),
  horse_number      integer not null,
  horse_name        text,
  decision_odds     numeric,
  odds_reference    numeric,
  odds_observed_at  timestamptz,
  p_model           numeric,
  p_low             numeric,
  p_market          numeric,
  p_fund            numeric,
  ev                numeric,
  model_rank        integer,
  market_rank       integer,
  recent_form       text,
  jockey            text,
  trainer           text,
  weight            numeric,
  draw              integer,
  age               integer,
  sex               text,
  career_starts     integer,
  career_wins       integer,
  career_places     integer,
  career_earnings   numeric,
  handicap_distance integer,
  primary key (snapshot_id, horse_number)
);

create table if not exists public.sg_settlements (
  snapshot_id         text primary key references public.sg_predictions(snapshot_id),
  settled_at          timestamptz not null,
  race_id             text,
  result_available    boolean not null,
  refunded            boolean not null default false,
  winners             jsonb,
  candidate_number    integer,
  candidate_won       boolean,
  candidate_dividend  numeric,
  simulated_bet       boolean not null default false,
  stake               numeric not null default 0,
  gross_return        numeric,
  profit              numeric,
  p_model_winner      numeric,
  p_market_winner     numeric,
  settlement_note     text
);

-- Immuabilité des photographies pré-course.
create or replace function public.sg_forbid_mutation()
returns trigger language plpgsql set search_path = '' as $$
begin
  raise exception 'Les prédictions Simple Gagnant sont immuables (table %, opération %).', tg_table_name, tg_op;
end;
$$;

drop trigger if exists trg_sg_predictions_immutable on public.sg_predictions;
create trigger trg_sg_predictions_immutable before update or delete on public.sg_predictions
  for each row execute function public.sg_forbid_mutation();

drop trigger if exists trg_sg_prediction_runners_immutable on public.sg_prediction_runners;
create trigger trg_sg_prediction_runners_immutable before update or delete on public.sg_prediction_runners
  for each row execute function public.sg_forbid_mutation();

alter table public.sg_predictions enable row level security;
alter table public.sg_prediction_runners enable row level security;
alter table public.sg_settlements enable row level security;

drop policy if exists "public read sg predictions" on public.sg_predictions;
create policy "public read sg predictions" on public.sg_predictions for select to anon, authenticated using (true);
drop policy if exists "public read sg prediction runners" on public.sg_prediction_runners;
create policy "public read sg prediction runners" on public.sg_prediction_runners for select to anon, authenticated using (true);
drop policy if exists "public read sg settlements" on public.sg_settlements;
create policy "public read sg settlements" on public.sg_settlements for select to anon, authenticated using (true);

create or replace view public.sg_journal with (security_invoker = true) as
select p.*, s.settled_at, s.result_available, s.refunded, s.winners, s.candidate_won, s.candidate_dividend,
       s.simulated_bet, s.stake, s.gross_return, s.profit, s.p_model_winner, s.p_market_winner, s.settlement_note
from public.sg_predictions p
left join public.sg_settlements s using (snapshot_id);
