"""Induction investment lab: decision-first interface, advanced maths, and provenance."""
from __future__ import annotations
import io
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from model import (PRESETS, Project, npv, validate, cash_flows, discounted_payback,
    break_even_savings, break_even_tariff, max_affordable_investment, simple_payback,
    gradient, hessian, taylor_report, tornado, elasticity, delta_method, make_sigma,
    monte_carlo_extended, replace_now_vs_wait, timing_break_even_k, calibrated_sds,
    sigma_g_horizon)

ROOT=Path(__file__).parent
st.set_page_config(page_title='Induction | Restaurant investment', page_icon='I', layout='wide', initial_sidebar_state='expanded')
st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"],.stApp{font-family:'DM Sans',sans-serif}
.block-container{max-width:1230px;padding-top:2.2rem;padding-bottom:4rem}
h1,h2,h3{font-family:'Manrope',sans-serif;letter-spacing:-.035em}
h1{font-size:2.25rem!important;font-weight:800!important}
h2{font-weight:700!important}
[data-testid="stMetric"]{padding:1.25rem;border:1px solid rgba(125,138,150,.28);border-radius:12px;background:rgba(126,138,150,.055)}
[data-testid="stMetricLabel"],[data-testid="stMetricValue"]{color:inherit!important}
[data-testid="stMetricValue"]{font-size:1.5rem!important;letter-spacing:-.04em}
[data-testid="stSidebar"]{border-right:1px solid rgba(125,138,150,.2)}
.hero-kicker{color:#548b82;font-size:.77rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase}
.hero-desc{opacity:.75;max-width:670px;font-size:1rem;line-height:1.65}
.stTabs [data-baseweb="tab"]{font-weight:600}
small{opacity:.7}
</style>''',unsafe_allow_html=True)

money=lambda x:f"CHF {float(x):,.0f}".replace(',','’')
num=lambda x:f"{float(x):,.0f}".replace(',','’')
@st.cache_data(show_spinner=False)
def tariffs_from_bytes(data):
    df=pd.read_csv(io.BytesIO(data),sep=None,engine='python')
    needed={'year','gemeindeName','gemeindeNummer','variable_tariff'}
    if not needed.issubset(df):
        raise ValueError('Expected columns: year, gemeindeName, gemeindeNummer, variable_tariff')
    for key in ('year','variable_tariff'):
        df[key]=pd.to_numeric(df[key].astype(str).str.replace(',','.'),errors='coerce')
    return df.dropna(subset=['year','variable_tariff']).sort_values('year')
@st.cache_data(show_spinner=False)
def yields_from_bytes(data):
    df=pd.read_csv(io.BytesIO(data),sep=None,engine='python')
    if {'year','yield_percent'}.issubset(df):
        frame=df[['year','yield_percent']].rename(columns={'yield_percent':'last'})
    elif {'year','last'}.issubset(df):
        frame=df[['year','last']]
    elif 'last' in df.columns:
        frame=df.iloc[:,[0,list(df.columns).index('last')]].copy();frame.columns=['year','last']
    else:
        raise ValueError('Expected year and yield_percent (or last) columns')
    frame['year']=pd.to_numeric(frame.year,errors='coerce')
    frame['last']=pd.to_numeric(frame['last'].astype(str).str.replace(',','.'),errors='coerce')
    return frame.dropna().sort_values('year')

with st.sidebar:
    st.markdown('### Data & assumptions')
    st.caption('C3 municipal benchmarks load automatically. Replace with your actual invoice values for investment decisions.')
    el_file=st.file_uploader('ElCom C3 tariff CSV',type='csv')
    snb_file=st.file_uploader('Historical SNB 10-year yield CSV',type='csv')
    st.divider()
    st.markdown('**Scope**')
    st.caption('Conventional electric hob → induction. This model does not cover gas-to-electric conversions.')
    st.caption('Amounts and energy savings are relative to the chosen baseline; tax and omitted costs are not automatically included.')
    st.divider()
    st.markdown('**Data status**')
    st.caption('ElCom C3: user-supplied 2020–2027 data. Ten-year Swiss government bond yields: user-supplied annual series (2020–2026), plus a separately documented one-day SNB observation. Original annual extraction not independently verified.')

st.markdown('<div class="hero-kicker">Energy investment · Switzerland</div>',unsafe_allow_html=True)
st.title('Is induction worth the investment?')
st.markdown('<p class="hero-desc">A decision tool for restaurants replacing a conventional electric cooking hob. Compare costs and savings, test what could change the decision, and inspect the mathematical evidence.</p>',unsafe_allow_html=True)

st.header('01 / Define the decision')
scenario=st.radio('What is your situation?',('Replacement already due · incremental cost','Replace a working hob today · full cost'),horizontal=True)
key='induction_incremental_sfoe' if scenario.startswith('Replacement already') else 'induction_replace_now_sfoe'
preset=PRESETS[key]

st.header('02 / Your restaurant')
st.caption('Edit the values below. Results update as soon as an input changes. Defaults are SFOE-inspired modelling benchmarks, not a supplier quotation.')
with st.container(border=True):
    a,b,c=st.columns(3,gap='large')
    with a:
        cost=st.number_input('Upfront equipment / incremental cost (CHF)',min_value=0.,value=float(preset['I0_gross']['value']),step=500.,key=f'cost_{key}')
        subsidy=st.number_input('Confirmed subsidy (CHF)',min_value=0.,max_value=float(cost),value=0.,step=100.,key=f'subsidy_{key}')
    with b:
        k=st.number_input('Year-one electricity savings (kWh)',min_value=1.,value=float(preset['k']['value']),step=500.,key=f'k_{key}')
        M=st.number_input('Additional annual maintenance (CHF)',min_value=0.,value=float(preset['M']['value']),step=50.,key=f'M_{key}')
    with c:
        T=st.number_input('New hob lifetime (years)',min_value=1,max_value=60,value=int(preset['T']['value']),key=f'T_{key}')
        decay=st.number_input('Annual decline in savings (%)',min_value=0.,max_value=50.,value=0.,step=.5,key=f'decay_{key}')/100

# ElCom: use the bundled C3 file by default; uploading another CSV overrides it.
elcom=None;hist_tariffs=None;sel_tariff=None
trace_tariff=None; y=None; place=None; tariff_description='Manual tariff assumption (no ElCom file loaded)'
try:
    bundled_elcom=ROOT/'data'/'tariffs_C3_clean.csv'
    elcom=tariffs_from_bytes(el_file.getvalue() if el_file else bundled_elcom.read_bytes())
    origin='User-uploaded C3 data' if el_file else 'Bundled user-supplied ElCom-derived C3 data'
    if elcom.empty:raise ValueError('No valid C3 rows')
    available_years=sorted(elcom.year.astype(int).unique(),reverse=True)
    # Never silently select a future tariff year.
    default_year=max((z for z in available_years if z<=2026),default=available_years[-1])
    y=st.selectbox('ElCom C3 tariff year',available_years,index=available_years.index(default_year))
    current=elcom[elcom.year==y]
    names=sorted(current.gemeindeName.dropna().astype(str).unique().tolist())
    place=st.selectbox('Municipality', ['National median']+names)
    if place=='National median':
        # Keep each municipality's influence equal, regardless of operator count.
        trace_tariff=(current.groupby(['year','gemeindeNummer'],as_index=False)
                      .agg(variable_tariff=('variable_tariff','median'),
                           operator_rows=('variable_tariff','size')))
        hist_tariffs=(elcom.groupby(['year','gemeindeNummer'],as_index=False)
                      .variable_tariff.median().groupby('year').variable_tariff.median().sort_index())
    else:
        trace_tariff=current.loc[current.gemeindeName==place].copy()
        hist_tariffs=(elcom.loc[elcom.gemeindeName==place]
                      .groupby('year').variable_tariff.median().sort_index())
    sel_tariff=float(trace_tariff.variable_tariff.median())
    tariff_description=f'{origin} · {place}, {y} · median of {len(trace_tariff):,} selected rows · Rp/kWh'
    if y>2026:st.info('2027 is a future tariff year in this dataset, not an observed realised 2027 electricity bill.')
except Exception as exc:
    st.warning(f'Could not load the C3 tariff dataset: {exc}. Enter a manual electricity tariff.')

# Annual SNB-style yield summary is bundled by default, with optional user upload.
# The annual summary represents the user's processed data, not authenticated raw SNB rows.
annual_source=ROOT/'data'/'yields_10y_annual.csv'
snb_annual=None
snb_origin='Bundled project-team annual 10-year Swiss Confederation yield summary'
try:
    raw_yields=pd.read_csv(io.BytesIO(snb_file.getvalue()) if snb_file else annual_source)
    if {'year','last'}.issubset(raw_yields.columns):
        snb_annual=raw_yields.copy()
        for field in ('year','mean','last','min','max','n_obs'):
            if field in snb_annual.columns:
                snb_annual[field]=pd.to_numeric(snb_annual[field],errors='coerce')
        snb_annual=snb_annual.dropna(subset=['year','last']).sort_values('year')
    else:
        snb_annual=yields_from_bytes(snb_file.getvalue() if snb_file else annual_source.read_bytes())
    if snb_annual.empty:raise ValueError('No usable years and bond yields')
    if snb_file:snb_origin='User-uploaded annual yield summary'
except Exception as exc:
    st.warning(f'Cannot load annual yield data: {exc}. Falling back to the documented one-day snapshot.')

snapshot=pd.read_csv(ROOT/'data'/'snb_verified_observation.csv')
rf_default=float(snapshot.iloc[-1].yield_percent)
trace_yield=snapshot.copy()
yield_description='Separately documented one-day SNB observation, 2026-11-06 (0.450%)'
hist_yields=None
selected_yield_year=None
selected_yield_measure=None
if snb_annual is not None:
    yield_years=sorted(snb_annual['year'].astype(int).unique(),reverse=True)
    selected_yield_year=st.selectbox('Ten-year Swiss Confederation yield year',yield_years,
                                     key='snb_annual_year')
    selected_yield_measure=st.radio('Annual yield statistic',('Last observation','Annual mean'),horizontal=True,
                                     help='Last is the final observation included in that year; mean averages that year’s observations. Both are percentages per annum.')
    value_col='last' if selected_yield_measure=='Last observation' else 'mean'
    if value_col not in snb_annual.columns:
        st.warning('Annual mean is unavailable in this file; using last observation instead.')
        value_col='last'
    trace_yield=snb_annual.loc[snb_annual['year'].astype(int)==selected_yield_year].copy()
    rf_default=float(trace_yield.iloc[-1][value_col])
    # Consistently use the selected annual statistic for annual SD calibration.
    hist_yields=snb_annual.groupby('year')[value_col].mean().sort_index()
    yield_description=(f'{snb_origin} · {selected_yield_year} · {value_col} = '
                       f'{rf_default:.3f}% (annual rate); provided summary, original extraction not verified')
    if selected_yield_year==2026:
        st.caption('2026 includes 217 daily observations in the supplied summary, not necessarily the complete calendar year. The annual last (0.505%) and the separate 6 November snapshot (0.450%) are different observations; do not treat them as identical.')

with st.container(border=True):
    a,b,c=st.columns(3,gap='large')
    with a:
        p_rp=st.number_input('Electricity tariff (Rp/kWh)',min_value=.01,value=float(sel_tariff if sel_tariff is not None else 25.),step=.25,format='%.2f',key=f'price_rp_{y}_{place}' if elcom is not None else 'price_rp_manual',help='Rp = Swiss centimes. This C3 price is a benchmark, not necessarily your avoidable marginal tariff.')
        share=st.slider('Share of tariff genuinely avoided (%)',1,100,80)/100
        st.caption(tariff_description if sel_tariff is not None and abs(p_rp-sel_tariff)<1e-7 else 'Manual tariff override (not the displayed source median)')
    with b:
        rf=st.number_input('Risk-free rate (%)',min_value=-10.,max_value=30.,value=rf_default,step=.1,key=f'risk_free_{selected_yield_year}_{selected_yield_measure}')
        spread=st.number_input('Additional risk spread (percentage points)',min_value=0.,max_value=30.,value=2.5,step=.25)
        st.caption('The 2.5pp spread is an assumption, not a verified risk premium.')
    with c:
        g=st.number_input('Nominal annual tariff growth (%)',min_value=-20.,max_value=30.,value=0.,step=.5)/100
        r=(rf+spread)/100
        st.metric('Nominal discount rate',f'{r:.2%}')
        st.caption('Nominal cash-flow growth is discounted using a nominal rate.')
P=Project(I0=cost-subsidy,k=k,p0=p_rp/100,T=int(T),M=M,delta=decay,share=share)
try:validate(P,g,r)
except ValueError as exc:st.error(exc);st.stop()

base=npv(P,g,r);be_k=break_even_savings(P,g,r);be_p=break_even_tariff(P,g,r)*100;pay=discounted_payback(P,g,r)
limit=max_affordable_investment(P,g,r)
st.header('03 / Investment result')
if base>0:st.success('Positive NPV under these assumptions — financially attractive relative to the chosen alternative.')
elif base<0:st.warning('Negative NPV under these assumptions — the modelled investment does not break even.')
else:st.info('The investment is approximately at break-even.')
c1,c2,c3,c4=st.columns(4)
c1.metric('Net present value',money(base));c2.metric('Break-even savings',f'{num(be_k)} kWh/year')
c3.metric('Break-even tariff',f'{be_p:.2f} Rp/kWh');c4.metric('Discounted payback',f'Year {pay}' if pay else 'Not reached')
st.caption(f'At the current inputs, initial annual savings must reach {num(be_k)} kWh to cover the modelled cost. The maximum affordable net investment is {money(limit)}. The SFOE 18,000 kWh figure is a monitored whole-kitchen average, not a guarantee for your premises.')

# Explicitly define baseline/alternative in session state without hidden changes.
current_inputs={'scenario':scenario,'cost':cost,'subsidy':subsidy,'k':k,'M':M,'T':T,'delta':decay,
                'tariff_rp':p_rp,'share':share,'risk_free':rf,'spread':spread,'growth':g,'discount_rate':r}
if 'baseline_inputs' not in st.session_state:st.session_state.baseline_inputs=None

st.header('04 / Explore your decision')
tabs=st.tabs(['Cash flows','Sensitivity','Compare scenarios','Uncertainty','Mathematics','Replacement timing','Validation','Sources & exports'])
with tabs[0]:
    years=np.arange(1,T+1,dtype=int);flows=cash_flows(P,g);pv=flows/(1+r)**years
    cumul=np.r_[-P.I0,-P.I0+np.cumsum(pv)]
    fig=go.Figure()
    fig.add_bar(x=np.r_[0,years],y=np.r_[-P.I0,pv],name='Discounted net cash flows',marker_color='#498579')
    fig.add_scatter(x=np.r_[0,years],y=cumul,name='Cumulative NPV',mode='lines',line=dict(color='#c38a55',width=3),yaxis='y2')
    fig.add_hline(y=0,line_color='#81918c',line_dash='dash')
    if pay:fig.add_vline(x=pay,line_dash='dot',line_color='#c38a55',annotation_text=f'Payback Y{pay}')
    fig.update_layout(template='plotly_white',height=420,xaxis_title='Year',yaxis_title='Discounted flow (CHF)',
        yaxis2=dict(overlaying='y',side='right',title='Cumulative NPV (CHF)'),legend=dict(orientation='h',y=-.25),margin=dict(t=30))
    st.plotly_chart(fig,use_container_width=True)
    cash_df=pd.DataFrame({'year':years,'year_one_equivalent_kWh':P.k*(1-P.delta)**(years-1),'nominal_cash_flow_CHF':flows,
        'discounted_cash_flow_CHF':pv,'cumulative_NPV_CHF':cumul[1:]})
    st.dataframe(cash_df,hide_index=True,use_container_width=True)
    st.download_button('Download annual cash flows',cash_df.to_csv(index=False).encode(),file_name='cash_flows.csv')

with tabs[1]:
    st.subheader('What drives the decision?')
    st.caption('The tornado uses the documented/explicit preset scenario ranges, not arbitrary percentage widths. Rankings are conditional on those ranges.')
    ranges={}
    for name,source in [('I0','I0_gross'),('k','k'),('T','T'),('M','M'),('share','share'),('delta','delta')]:
        lo,hi=preset[source]['range']
        if name=='I0':lo=max(0,lo-subsidy);hi=max(0,hi-subsidy)
        if name=='k':lo=max(1.,lo)
        if name=='T':lo,hi=int(lo),int(hi)
        if name=='share':lo=max(.001,lo)
        ranges[name]=(lo,hi)
    if hist_tariffs is not None and len(hist_tariffs)>4:
        ranges['p0']=tuple(np.quantile(hist_tariffs,[.1,.9])/100)
    else:ranges['p0']=(P.p0*.75,P.p0*1.25)  # Explicit scenario only
    ranges['g']=(max(-.95,g-.02),g+.02)
    ranges['r']=(max(-.95,r-.01),r+.01)
    # Model tornado accepts arbitrary project fields, including delta.
    sensitivity=tornado(P,g,r,ranges)
    rows=[]
    for name,lo,hi in sensitivity:
        rows.append({'Input':name,'Low-case NPV (CHF)':lo,'High-case NPV (CHF)':hi,'NPV swing (CHF)':abs(hi-lo),
                     'Range':f'{ranges[name][0]:.4g} → {ranges[name][1]:.4g}'})
    sdf=pd.DataFrame(rows)
    fig=go.Figure(go.Bar(x=sdf.sort_values('NPV swing (CHF)')['NPV swing (CHF)'],
                        y=sdf.sort_values('NPV swing (CHF)')['Input'],orientation='h',marker_color='#498579'))
    fig.update_layout(template='plotly_white',height=390,xaxis_title='NPV swing (CHF)',yaxis_title='',margin=dict(t=20))
    st.plotly_chart(fig,use_container_width=True)
    st.dataframe(sdf,hide_index=True,use_container_width=True)
    st.caption('Tariff range: scenario ±25% unless enough uploaded historical municipal/national data allow an empirical p10–p90. Other ranges: inspect model.py PRESETS; g ±2pp and r ±1pp remain stress-test assumptions.')
    st.subheader('Dimensionless elasticities')
    ee=elasticity(P,g,r)
    if ee:
        edf=pd.DataFrame([{'Input':x,'Elasticity':v,'Absolute elasticity':abs(v)} for x,v in ee.items()]).sort_values('Absolute elasticity',ascending=False)
        st.dataframe(edf,hide_index=True,use_container_width=True)
        st.caption('Elasticity = percentage change in NPV for a 1% change in the input locally. Near NPV=0, elasticities become unstable; zero-valued input elasticities are omitted.')
    else:st.info('Elasticities are unstable close to NPV = 0; use absolute derivatives instead.')
    st.caption('Important: the least certain restaurant-specific quantities are generally equipment cost, actual energy savings and avoidable tariff share. The ranking is not a measure of statistical importance.')

with tabs[2]:
    st.subheader('Baseline versus alternative')
    if st.button('Save current inputs as baseline'):
        st.session_state.baseline_inputs=dict(current_inputs)
    baseline=st.session_state.baseline_inputs
    if baseline is None:
        st.info('Save the current inputs as your baseline. Then change the inputs above and compare them against that saved case.')
    else:
        Q=Project(I0=baseline['cost']-baseline['subsidy'],k=baseline['k'],p0=baseline['tariff_rp']/100,
                  T=int(baseline['T']),M=baseline['M'],delta=baseline['delta'],share=baseline['share'])
        gb=baseline['growth'];rb=baseline['discount_rate']
        bn=npv(Q,gb,rb);bp=discounted_payback(Q,gb,rb);bk=break_even_savings(Q,gb,rb)
        table=pd.DataFrame({'Measure':['Net present value (CHF)','Discounted payback (years)','Break-even savings (kWh)'],
            'Baseline':[bn,float(bp) if bp is not None else np.nan,bk],
            'Current alternative':[base,float(pay) if pay is not None else np.nan,be_k]})
        table['Change (alternative − baseline)']=table['Current alternative']-table['Baseline']
        st.dataframe(table,hide_index=True,use_container_width=True)
        st.caption('Payback is unavailable if the investment is not recovered during its lifetime. Changes are evaluated using each case’s own inputs; they are not causal estimates.')
        st.download_button('Download comparison',table.to_csv(index=False).encode(),file_name='baseline_comparison.csv')

with tabs[3]:
    st.subheader('Simulation with uncertain costs and savings')
    st.caption('Normal rate scenarios and independent lognormal cost/savings scenarios; these are not observed probability distributions unless supported by data. The delta method below still treats only growth and discount-rate uncertainty.')
    cal=calibrated_sds(hist_tariffs.values if hist_tariffs is not None else None,
                      hist_yields.values if hist_yields is not None else None,int(T))
    if 'sd_g' in cal:st.caption(f'Historical growth calibration available ({cal["tariff_observations"]} annual observations; independent-year assumption): {cal["sd_g"]:.2%}.')
    if 'sd_r' in cal:st.caption(f'Historical rate calibration available ({cal["yield_observations"]} annual observations): {cal["sd_r"]:.2%}.')
    a,b,c=st.columns(3)
    sdg=a.number_input('SD: horizon-average tariff growth (pp)',0.,30.,round(100*cal.get('sd_g',.02),3),.1)/100
    sdr=b.number_input('SD: discount rate (pp)',0.,30.,round(100*cal.get('sd_r',.01),3),.1)/100
    rho=c.slider('Growth–rate correlation',-1.,1.,.0,.1)
    a,b,c=st.columns(3)
    sdk=a.slider('Savings relative SD (%)',0,100,25)/100
    sdi=b.slider('Investment cost relative SD (%)',0,100,20)/100
    sds=c.slider('Avoidable-share absolute SD (pp)',0,50,10)/100
    N=st.select_slider('Simulation draws',options=[2000,5000,10000,20000,50000,100000],value=10000)
    Sigma=make_sigma(sdg,sdr,rho)
    mc=monte_carlo_extended(P,g,r,Sigma,n=N,sd_k_frac=sdk,sd_cost_frac=sdi,sd_share=sds)
    mm=delta_method(P,g,r,Sigma)
    a,b,c=st.columns(3)
    a.metric('Estimated P(NPV > 0)',f'{mc["p_positive"]:.2%}')
    b.metric('Wilson 95% interval',f'{mc["ci_low"]:.2%} – {mc["ci_high"]:.2%}')
    c.metric('Mean simulated NPV',money(mc['mean']))
    if mc['p_positive']==1:st.info(f'No losses in {N:,} draws. Do not report certainty: the simulated success probability has a 95% lower confidence bound of {mc["ci_low"]:.2%}.')
    fig=go.Figure(go.Histogram(x=mc['samples'],nbinsx=55,marker_color='#498579'))
    fig.add_vline(x=0,line_dash='dash',line_color='#ae7258')
    fig.update_layout(template='plotly_white',height=330,xaxis_title='NPV (CHF)',yaxis_title='Draw count',margin=dict(t=20))
    st.plotly_chart(fig,use_container_width=True)
    idx=np.unique(np.linspace(99,N-1,90,dtype=int))
    fig=go.Figure(go.Scatter(x=idx+1,y=mc['convergence'][idx],mode='lines',line=dict(color='#498579')))
    fig.update_layout(template='plotly_white',height=260,yaxis_title='Estimated P(NPV > 0)',xaxis_title='Draws',margin=dict(t=20))
    st.plotly_chart(fig,use_container_width=True)
    st.caption(f'Clipped unrealistic growth/discount-rate draws: {mc["clipped_share"]:.2%}; estimated NPV 5th–95th percentile {money(mc["p05"])} to {money(mc["p95"])}.')
    st.markdown('**Growth–rate correlation stress test**')
    correlations=[-0.6,0.0,0.6]
    compare=[]
    for corr in correlations:
        cm=monte_carlo_extended(P,g,r,make_sigma(sdg,sdr,corr),n=min(20000,N),
             seed=42,sd_k_frac=sdk,sd_cost_frac=sdi,sd_share=sds)
        compare.append({'Correlation':corr,'P(NPV>0)':cm['p_positive'],'Mean NPV':cm['mean']})
    st.dataframe(pd.DataFrame(compare),hide_index=True,use_container_width=True)
    st.caption(f'Delta method for rate/growth-only uncertainty: mean {money(mm["mean"])} vs NPV at the means {money(mm["npv_at_mean"])}; their gap illustrates the second-order curvature correction (related to Jensen’s inequality). It is not directly comparable to the extended simulation with stochastic equipment cost and kWh.')
    st.markdown('**Stress threshold for at least 90% simulated success**')
    st.caption('Reduce expected annual savings, holding the entered relative uncertainty assumptions fixed. Binary search locates an illustrative threshold, conditional on this simulation scenario.')
    def stress_prob(k_candidate):
        obj=replace(P,k=max(k_candidate,1e-5))
        return monte_carlo_extended(obj,g,r,Sigma,n=min(N,10000),seed=87,
                 sd_k_frac=sdk,sd_cost_frac=sdi,sd_share=sds)['p_positive']
    lo,hi=1e-5,max(P.k,10.)
    if stress_prob(hi)<.9:
        while stress_prob(hi)<.9 and hi<1e8:hi*=2
    if stress_prob(hi)>=.9:
        for _ in range(22):
            mid=(lo+hi)/2
            if stress_prob(mid)>=.9:hi=mid
            else:lo=mid
        st.metric('Approximate 90% success savings threshold',f'{num(hi)} kWh/year')
    else:st.warning('No 90% threshold found in the search range.')
    st.caption('Monte Carlo confidence bounds quantify finite simulation sampling error, not uncertainty in the scenario distributions or assumptions.')

with tabs[4]:
    st.subheader('Analytical gradient, Hessian and Taylor expansion')
    st.latex(r'NPV=-I_0+\sum_{t=1}^{T}\frac{k(1-\delta)^{t-1}p_0s(1+g)^{t-1}-M}{(1+r)^t}')
    st.dataframe(pd.DataFrame({'Derivative':['∂NPV/∂g','∂NPV/∂r'],'CHF per rate unit':gradient(P,g,r)}),hide_index=True)
    st.write('Hessian');st.dataframe(pd.DataFrame(hessian(P,g,r),index=['g','r'],columns=['g','r']))
    a,b=st.columns(2)
    dg=a.slider('Growth perturbation (pp)',-10.,10.,3.,.5)/100
    dr=b.slider('Rate perturbation (pp)',-5.,5.,1.,.5)/100
    try:
        results=taylor_report(P,g,r,dg,dr)
        st.dataframe(pd.DataFrame([{'Method':key,'Value':value} for key,value in results.items()]),hide_index=True)
    except ValueError as exc:st.warning(str(exc))
    st.caption('Taylor approximations are local. The main decision uses exact annual discounted cash flows.')

with tabs[5]:
    st.subheader('Replace the working hob now, or wait?')
    if key=='induction_incremental_sfoe':
        st.info('Timing requires the full purchase price, not incremental cost. Choose “Replace a working hob today” above.')
    else:
        n=st.slider('Remaining useful life of your existing hob (years)',0,T,min(5,T),1)
        diff=replace_now_vs_wait(P,g,r,n)['now_minus_wait']
        cutoff=timing_break_even_k(P,g,r,n)
        st.metric('PV advantage of replacing now',money(diff))
        st.metric('Minimum annual savings to make NOW preferable',f'{num(cutoff)} kWh' if cutoff is not None else 'No non-negative threshold')
        tt=pd.DataFrame([replace_now_vs_wait(P,g,r,i) for i in range(T+1)])
        fig=go.Figure(go.Scatter(x=tt.n,y=tt.now_minus_wait,mode='lines+markers',line=dict(color='#498579')))
        fig.add_hline(y=0,line_dash='dash',line_color='#ae7258')
        fig.update_layout(template='plotly_white',height=320,xaxis_title='Years remaining on old hob',yaxis_title='Now minus wait (CHF)',margin=dict(t=20))
        st.plotly_chart(fig,use_container_width=True)
        st.caption('The threshold uses bisection (numerical root-finding). The common-horizon timing model assumes no salvage on the old hob, equal future purchase prices and straight-line terminal residual value.')

with tabs[6]:
    st.subheader('Live model validation')
    from model import _A,_B,monte_carlo,check_covariance
    closed=-P.I0+P.k*P.p0*P.share*_B(P,g,r)-P.M*_A(P,r)
    eps=1e-5
    grads=gradient(P,g,r);H=hessian(P,g,r)
    fg=(npv(P,g+eps,r)-npv(P,g-eps,r))/(2*eps)
    fr=(npv(P,g,r+eps)-npv(P,g,r-eps))/(2*eps)
    hgg=(npv(P,g+eps,r)-2*npv(P,g,r)+npv(P,g-eps,r))/eps**2
    hrr=(npv(P,g,r+eps)-2*npv(P,g,r)+npv(P,g,r-eps))/eps**2
    hgr=(npv(P,g+eps,r+eps)-npv(P,g+eps,r-eps)-npv(P,g-eps,r+eps)+npv(P,g-eps,r-eps))/(4*eps**2)
    zero_mc=monte_carlo(P,g,r,make_sigma(0,0,0),n=10,seed=7)
    tests=[('NPV agrees with discounted-annuity identity',abs(closed-base)<1e-7,abs(closed-base)),
           ('Gradient matches central finite differences',np.max(abs(grads-[fg,fr]))<.1,np.max(abs(grads-[fg,fr]))),
           ('Hessian matches finite differences',np.max(abs(H-[[hgg,hgr],[hgr,hrr]]))<10,np.max(abs(H-[[hgg,hgr],[hgr,hrr]]))),
           ('Break-even kWh yields zero NPV',abs(npv(replace(P,k=be_k),g,r))<1e-6,abs(npv(replace(P,k=be_k),g,r))),
           ('Break-even tariff yields zero NPV',abs(npv(replace(P,p0=be_p/100),g,r))<1e-6,abs(npv(replace(P,p0=be_p/100),g,r))),
           ('Zero-uncertainty Monte Carlo equals NPV',abs(zero_mc['mean']-base)<1e-7,abs(zero_mc['mean']-base))]
    try:validate(replace(P,T=0),g,r);bad_T=False
    except ValueError:bad_T=True
    try:make_sigma(.02,.01,1.2);bad_corr=False
    except ValueError:bad_corr=True
    tests.extend([('Reject invalid T=0',bad_T,0),('Reject invalid correlation ρ=1.2',bad_corr,0)])
    st.dataframe(pd.DataFrame([{'Check':label,'Status':'PASS' if passed else 'FAIL','Absolute error':float(err)} for label,passed,err in tests]),hide_index=True,use_container_width=True)
    st.caption('Finite differences are numerically approximate. The companion tests/test_model.py provides repeatable unit tests.')

with tabs[7]:
    st.subheader('Provenance: trace the values')
    st.markdown('**Electricity price**')
    st.write(tariff_description)
    st.write('Applied tariff:',f'{p_rp:.2f} Rp/kWh; avoidable fraction: {share:.0%}; effective marginal avoided cost: {p_rp*share:.2f} Rp/kWh')
    if trace_tariff is not None:
        st.write(f'Source median (before override): {sel_tariff:.3f} Rp/kWh; source row count: {len(trace_tariff):,}')
        st.dataframe(trace_tariff,hide_index=True,use_container_width=True)
        st.download_button('Download source rows used',trace_tariff.to_csv(index=False).encode(),
                           file_name=f'elcom_trace_{y}.csv',mime='text/csv')
        if not el_file:
            annual=pd.read_csv(ROOT/'data'/'tariffs_C3_summary_by_year.csv')
            st.markdown('**Provided annual C3 summary (national municipality-level medians)**')
            st.dataframe(annual,hide_index=True,use_container_width=True)
            st.caption('2020–2027 data and annual summary provided by the project team. Original provider extraction and download dates have not been independently authenticated.')
    else:st.warning('No ElCom rows available. Check the original CSV or enter your own invoice tariff.')
    st.markdown('**Risk-free rate**');st.write(yield_description)
    st.dataframe(trace_yield,hide_index=True,use_container_width=True)
    if snb_annual is not None:
        st.markdown('**Bundled annual 10-year Confederation yields (2020–2026)**')
        st.dataframe(snb_annual,hide_index=True,use_container_width=True)
        st.download_button('Download source annual yield summary',snb_annual.to_csv(index=False).encode(),file_name='yield_trace_annual.csv',mime='text/csv')
        st.caption('Unit for mean, last, min and max: percent per annum (not decimals). n_obs counts underlying observations. Data supplied by the project team; underlying raw daily SNB observations and retrieval date were not provided. The 2026 row is partial.')
    st.write(f'Applied risk-free rate: {rf:.3f}%; unverified spread: {spread:.2f}pp; nominal discount rate: {r:.2%}.')
    st.markdown('**Year-one energy saving / investment cost**')
    st.write(f'{num(k)} kWh/year and {money(cost)} gross investment, entered by the user. SFOE research benchmarks are not restaurant-specific measurements or quotations.')
    st.markdown('**References and source metadata**')
    st.markdown('- ElCom. (n.d.). *Tariff data and visualisations*. https://www.elcom.admin.ch/en/tariff-data-and-visualisations — C3 tariffs; unit Rp/kWh; 2020–2027 cleaned local CSV **provided by the project team**; original extraction and retrieval dates unverified.\n- Swiss National Bank. (n.d.). *SNB Data Portal*. https://data.snb.ch/en — project-team supplied annual 10-year Swiss Confederation yield summary (2020–2026); original extraction date and daily observations not independently verified. Separately documented spot observation: 0.450% on 6 November 2026.\n- Swiss Federal Office of Energy. (2022). *Potenzial und Massnahmen zur Steigerung der Stromeffizienz bis 2025*, measure 31a — monitored 18,000 kWh/year energy savings and modelled price assumptions as described in PRESETS.')
    summary=dict(current_inputs,NPV_CHF=base,break_even_savings_kWh=be_k,
                 break_even_tariff_Rp_kWh=be_p,discounted_payback_year=pay,
                 tariff_data_origin=tariff_description,rf_data_origin=yield_description)
    st.download_button('Download inputs and outputs (CSV)',pd.DataFrame([summary]).to_csv(index=False).encode(),file_name='induction_results.csv',mime='text/csv')
    prompt=('Assess the economics of replacing a conventional ELECTRIC restaurant hob with induction. '
            'No gas conversion is considered. Inputs are scenario assumptions unless their source is verified.\n'
            +json.dumps(summary,indent=2,default=str)
            +'\nRecompute the NPV from year-by-year discounted cash flows. Show formulas, every numerical step, break-even energy saving, decision limitations, and full provenance. Do not invent supplier quotations or subsidy eligibility. Please identify which assumptions drive the result.')
    st.download_button('Export as AI-comparison prompt',prompt.encode('utf-8'),file_name='ai_comparison_prompt.txt',mime='text/plain')
    st.caption('For the required AI comparison, record the AI model, test date, complete answer and discrepancies against the tool. Do not claim the comparison was performed until documented.')

st.divider()
st.caption('Mathematical decision support, not a supplier quotation or promise of savings. All calculations are nominal and exclude unspecified expenses. Designed for conventional electric-to-induction replacements only.')
