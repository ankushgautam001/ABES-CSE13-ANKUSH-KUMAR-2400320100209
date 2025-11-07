
const API_KEY = "f4b4b761c193a634940eeb55e9df7466"; // 🔑 Replace with your OpenWeatherMap API key
const units = "metric"; // "imperial" for °F

const searchForm = document.getElementById("searchForm");
const cityInput = document.getElementById("cityInput");
const geoBtn = document.getElementById("geoBtn");
const statusEl = document.getElementById("status");
const weatherCard = document.getElementById("weatherCard");
const cityName = document.getElementById("cityName");
const desc = document.getElementById("desc");
const icon = document.getElementById("icon");
const temp = document.getElementById("temp");
const feels = document.getElementById("feels");
const hum = document.getElementById("hum");
const wind = document.getElementById("wind");

function showStatus(msg, isError=false){
  statusEl.textContent = msg;
  statusEl.classList.remove("hidden");
  statusEl.style.color = isError ? "#b91c1c" : "";
}
function hideStatus(){statusEl.classList.add("hidden");}
function showCard(){weatherCard.classList.remove("hidden");}
function hideCard(){weatherCard.classList.add("hidden");}

function capitalize(s=""){return s.replace(/\b\w/g,c=>c.toUpperCase());}

async function getWeatherByCity(city){
  const url = `https://api.openweathermap.org/data/2.5/weather?q=${encodeURIComponent(city)}&appid=${API_KEY}&units=${units}`;
  const res = await fetch(url);
  if(!res.ok) throw new Error("City not found");
  return res.json();
}

async function getWeatherByCoords(lat,lon){
  const url = `https://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lon}&appid=${API_KEY}&units=${units}`;
  const res = await fetch(url);
  if(!res.ok) throw new Error("Location not found");
  return res.json();
}

function renderWeather(data){
  const name = `${data.name}${data.sys?.country ? ", "+data.sys.country : ""}`;
  const w = data.weather?.[0];
  cityName.textContent = name;
  desc.textContent = w ? capitalize(w.description) : "—";
  icon.src = w ? `https://openweathermap.org/img/wn/${w.icon}@2x.png` : "";
  icon.alt = w?.description || "";
  temp.textContent = `${Math.round(data.main.temp)} °${units==="metric"?"C":"F"}`;
  feels.textContent = `${Math.round(data.main.feels_like)} °${units==="metric"?"C":"F"}`;
  hum.textContent = `${data.main.humidity}%`;
  wind.textContent = `${data.wind.speed} ${units==="metric"?"m/s":"mph"}`;
  showCard();
}

searchForm.addEventListener("submit",async e=>{
  e.preventDefault();
  const city = cityInput.value.trim();
  if(!city){showStatus("Enter a city name",true);return;}
  hideCard();
  showStatus("Loading...");
  try{
    const data = await getWeatherByCity(city);
    hideStatus(); renderWeather(data);
  }catch(err){
    hideCard(); showStatus(err.message,true);
  }
});

geoBtn.addEventListener("click",()=>{
  if(!navigator.geolocation){
    showStatus("Geolocation not supported",true);return;
  }
  hideCard(); showStatus("Fetching location...");
  navigator.geolocation.getCurrentPosition(async pos=>{
    try{
      showStatus("Loading weather...");
      const data = await getWeatherByCoords(pos.coords.latitude,pos.coords.longitude);
      hideStatus(); renderWeather(data);
    }catch(err){
      showStatus(err.message,true);
    }
  },err=>{
    showStatus("Permission denied or unavailable",true);
  });
});
