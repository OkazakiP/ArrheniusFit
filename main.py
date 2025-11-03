import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import pandas as pd



class Model:
    """ある温度における特性の時間変化を予測する.

    特性Pが化学種濃度C1, C2, .., Cnの関数とする．
    また反応はm段階あり，各反応における反応速度定数kiはアレニウスの式で表されるとする．
    この時，各化学種の濃度Ciの時間変化はm個の反応速度定数を用いて記述できる．
    これらを式で表すと以下の通り．

    特性P: f(C1, C2, ..., Cn)
    反応速度: (dC1/dt, dC2/dt, ..., dCn/dt) = (..., g_n(vector(C), k1, k2, .., km))
    アレニウスの式: ki = Ai * exp(-Eai/RT)

    この条件下，反応速度の式を連立常微分方程式として`scipy.integrate.solve_ivp`で解き，
    得られた解C1, C2, ..., Cnを用いて特性Pを予測する．
    
    Attributes
    ----------
    model_reaction: callable
        Right-hand side of the system, (dC1/dt, dC2/dt, ..., dCn/dt).
        The calling signature is `model_reaction(t, C)`, where t
        is a scaler and `C` is an ndarray.
    n_reaction_step: int
        The number of reactions. k1, k2, ..., k_n.
    model_property: callable
        Function that predicts chemical property.
    n_species: int
        The number of chemical species. C1, C2, ..., Cn.
    params_reaction: array-like
        Arguments for reaction method.
    params_property: array-like
        Arguments for property method.
    
    Methods
    -------
    fit(data)
        Fit parameters of `reaction` and `property` method.
    reaction(t, T, *params)
        Predict molecular concentration at the time in the temperature.
    property(C, *params)
        Predict chemical property in the given molecular concentration.
    predict(t, T, params_reaction=None, params_property=None)
        Predict chemical property at the time in the temperature.
    life(prop, T, params_reaction=None, params_property=None)
        Predict how the property would take to reach `prop` in the temperature.
    """
    def __init__(self, model_reaction, n_reaction_step, model_property, n_species):
        """
        Parameters
        ----------
        model_reaction: callable
            Right-hand side of the system, (dC1/dt, dC2/dt, ..., dCn/dt).
            The calling signature is `model_reaction(t, C)`, where t
            is a scaler and `C` is an ndarray.
        n_reaction_step: int
            The number of reactions. k1, k2, ..., k_n.
        model_property: callable
            Function that predicts chemical property.
        n_species: int
            The number of chemical species. C1, C2, ..., Cn.
        """
        self.model_reaction = model_reaction
        self.n_reaction_step = n_reaction_step
        self.model_property = model_property
        self.n_species = n_species

    def fit(self, data: pd.DataFrame):
        """Fit parameters of `reaction` and `property method.
        
        Parameters
        ----------
        data: pd.DataFrame
            Measurement data.
        """
        def total_loss(params):
            C0 = params[:self.n_species]
            A = params[self.n_species:self.n_species+self.n_reaction_step]
            Ea = params[self.n_species+self.n_reaction_step: self.n_species+self.n_reaction_step*2]
            p = params[self.n_species+self.n_reaction_step*2:]
            loss = 0
            for T, group in data.groupby('T'):
                t = group['time']
                t_data = np.logspace(0, np.ceil(np.log10(t.max())), 1000)
                t_data = np.concat([t.unique(), t_data])
                t_data = np.sort(np.array(set(t_data)))
                obs = group.set_index('time')['observed'].sort_index()
                sol = solve_ivp(
                    self.model_reaction, [t_data[0], t_data[-1]], C0,
                    t_eval=t_data, args=(A, Ea, T)
                )
                C_pred = sol.y
                P_pred = (
                    pd.Series(self.model_property(C_pred), index=t_data)
                    .rename_axis('time', axis=0)
                )
                loss += obs.sub(P_pred, fill_value=0.).pipe(lambda s: s**2).sum()
            return loss

        res = minimize(total_loss, x0=[1.0, 4000.0])



if __name__=='__main__':
    R = 8.31

    # 例：温度ごとに異なる時刻点
    T_conditions = [300, 350, 400, 450]
    t_data_sets = [np.linspace(0, 10, 40),
                np.linspace(0, 5, 30),
                np.linspace(0, 3, 20),
                np.linspace(0, 2, 15)]

    # 真のパラメータ
    A_true, Ea_true = 2.0, 5000.0

    # 真のモデル
    def model_rhs(t, C, A, Ea, T):
        k = A * np.exp(-Ea / (R*T))
        return -k*C

    # データ生成
    C0 = 1.0
    data = []
    for T, t_data in zip(T_conditions, t_data_sets):
        sol = solve_ivp(model_rhs, [t_data[0], t_data[-1]], [C0], t_eval=t_data, args=(A_true, Ea_true, T))
        C_true = sol.y[0]
        C_obs = C_true + 0.05 * np.random.randn(len(C_true))
        data.append((T, t_data, C_obs))

    # 一括Loss関数
    def total_loss(params):
        A, Ea = params
        loss = 0
        for T, t_data, C_obs in data:
            sol = solve_ivp(model_rhs, [t_data[0], t_data[-1]], [C0], t_eval=t_data, args=(A, Ea, T))
            C_pred = sol.y[0]
            w = 1.0 / len(t_data)  # データ点数で正規化
            loss += w * np.sum((C_pred - C_obs)**2)
        return loss

    res = minimize(total_loss, x0=[1.0, 4000.0])
    print(f"推定値: A={res.x[0]:.3f}, Ea={res.x[1]:.2f}")
