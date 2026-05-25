import axios from "axios";

const BASE_URL = "http://127.0.0.1:8000";
const SEARCH_PATH = "/songs";

function extractSongList(payload) {
  if (Array.isArray(payload)) return payload;
  if (Array.isArray(payload?.songs)) return payload.songs;
  if (Array.isArray(payload?.results)) return payload.results;
  if (Array.isArray(payload?.items)) return payload.items;
  if (Array.isArray(payload?.data)) return payload.data;
  return [];
}

function normalizeSong(raw, index) {
  const title = raw?.title ?? raw?.song_title ?? raw?.name ?? raw?.track ?? `제목 미상 ${index + 1}`;
  const artist = raw?.artist ?? raw?.singer ?? raw?.artist_name ?? "아티스트 정보 없음";
  const album = raw?.album ?? raw?.album_name ?? "";
  const duration = raw?.duration ?? raw?.play_time ?? raw?.length ?? "";
  const key = raw?.key ?? raw?.song_key ?? raw?.pitch_key ?? "";
  const coverUrl = raw?.cover_url ?? raw?.thumbnail ?? raw?.image ?? "";
  const id = raw?.id ?? raw?.song_id ?? `${title}-${artist}-${index}`;
  const midiMin = raw?.midi_min ?? raw?.midiMin ?? null;
  const midiMedian = raw?.midi_median ?? raw?.midiMedian ?? null;
  const midiMax = raw?.midi_max ?? raw?.midiMax ?? null;
  const rmsMean = raw?.rms_mean ?? raw?.rmsMean ?? null;
  const rmsStd = raw?.rms_std ?? raw?.rmsStd ?? null;

  return {
    id,
    title,
    artist,
    album,
    duration,
    key,
    coverUrl,
    midiMin,
    midiMedian,
    midiMax,
    rmsMean,
    rmsStd,
  };
}

export async function getSongs(keyword) {
  const q = keyword?.trim();
  if (!q) return [];

  const res = await axios.get(`${BASE_URL}${SEARCH_PATH}`, {
    params: { q, query: q, keyword: q },
  });

  const list = extractSongList(res.data);
  return list.map(normalizeSong);
}

export async function getSongCatalog() {
  const res = await axios.get(`${BASE_URL}${SEARCH_PATH}`);
  const list = extractSongList(res.data);
  return list.map(normalizeSong);
}
