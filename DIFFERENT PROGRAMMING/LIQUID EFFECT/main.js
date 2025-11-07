import LiquidBackground from 'https://cdn.jsdelivr.net/npm/threejs-components@0.0.22/build/backgrounds/liquid1.min.js'

const app = LiquidBackground(document.getElementById('canvas'))

app.loadImage('')

app.liquidPlan.material.metalness=0.75
app.liquidPlan.material.roughness=0.25
app.liquidPlan.uniforms.displacemnetScale.value=5

app.setRain(false)