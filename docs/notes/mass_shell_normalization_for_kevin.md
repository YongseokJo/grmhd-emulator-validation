# Note for Kevin: evaluator mass-shell normalization

In `Fall26/eval/evaluate_forecast_ds.py` (`main@ebcc218`), fig 5
(`evaluate_shell_profiles`) and Table 2 (`evaluate_physics_table`) divide the
bare angular sum for M(r) by `gdet.shape[-2]*gdet.shape[-1]`. The geometry stores
a singleton phi axis, so this divides by Ntheta rather than physical Ntheta*Nphi.
The reported M(r) is therefore inflated by Nphi (96 for rc300).

Affected outputs are fig 5's mass truth/forecast profiles and signed-log mass
error panel, and Table 2's signed-log mass error column. Because signed log1p
is nonlinear, the error cannot be repaired by simply dividing it by Nphi.
Energy/angular-momentum profiles already use the physical angular count.
`PhysicsLoss3D._mass_shell` also uses the correct physical count; this evaluator
fix does not change the training loss or checkpoint weights.

The benchmark now divides these production mass profiles by the state's last
two spatial dimensions. The original evaluator expression remains available
only in the explicit legacy regression mode, so the source regression is still
faithful. No message has been sent; this note is for the owner to pass along.
