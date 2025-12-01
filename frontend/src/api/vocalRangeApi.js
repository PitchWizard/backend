// frontend/src/api/vocalRangeApi.js
import axios from "axios";

const BASE_URL = "http://127.0.0.1:8000"; // FastAPI 주소

export default async function saveVocalRange(payload) {
  // payload 예:
  // {
  //   user_id: 1,
  //   midi_min: 48,
  //   midi_median: 58.5,
  //   midi_max: 69,
  //   low_note: "C3",
  //   high_note: "A4",
  //   avg_rms: null
  // }
  const res = await axios.post(`${BASE_URL}/vocal-range`, payload);
  return res.data;
}
