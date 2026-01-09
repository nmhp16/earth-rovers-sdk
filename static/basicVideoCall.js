/*
 *  These procedures use Agora Video Call SDK for Web to enable local and remote
 *  users to join and leave a Video Call channel managed by Agora Platform.
 */

/*
 *  Create an {@link https://docs.agora.io/en/Video/API%20Reference/web_ng/interfaces/iagorartcclient.html|AgoraRTCClient} instance.
 *
 * @param {string} mode - The {@link https://docs.agora.io/en/Voice/API%20Reference/web_ng/interfaces/clientconfig.html#mode| streaming algorithm} used by Agora SDK.
 * @param  {string} codec - The {@link https://docs.agora.io/en/Voice/API%20Reference/web_ng/interfaces/clientconfig.html#codec| client codec} used by the browser.
 */
var client;

/*
 * Clear the video and audio tracks used by `client` on initiation.
 */
var localTracks = {
  videoTrack: null,
  audioTrack: null,
};

/*
 * On initiation no users are connected.
 */
var remoteUsers = {};

/*
 * On initiation. `client` is not attached to any project or channel for any specific user.
 */
var options = {
  appid: null,
  channel: null,
  uid: null,
  token: null,
};
var pendingSubscriptions = new Set();
var incompatibleSubscriptions = new Map(); // key -> timestamp (ms)

if (typeof RTCPeerConnection !== 'undefined' && RTCPeerConnection.prototype && RTCPeerConnection.prototype.getStats) {
  const _origGetStats = RTCPeerConnection.prototype.getStats;
  RTCPeerConnection.prototype.getStats = function(selectors) {
    try {
      return _origGetStats.call(this, selectors);
    } catch (err) {
      return Promise.resolve(new Map());
    }
  };
}

// you can find all the agora preset video profiles here https://docs.agora.io/en/Voice/API%20Reference/web_ng/globals.html#videoencoderconfigurationpreset
var videoProfiles = [
  {
    label: "360p_7",
    detail: "480×360, 15fps, 320Kbps",
    value: "360p_7",
  },
  {
    label: "360p_8",
    detail: "480×360, 30fps, 490Kbps",
    value: "360p_8",
  },
  {
    label: "480p_1",
    detail: "640×480, 15fps, 500Kbps",
    value: "480p_1",
  },
  {
    label: "480p_2",
    detail: "640×480, 30fps, 1000Kbps",
    value: "480p_2",
  },
  {
    label: "720p_1",
    detail: "1280×720, 15fps, 1130Kbps",
    value: "720p_1",
  },
  {
    label: "720p_2",
    detail: "1280×720, 30fps, 2000Kbps",
    value: "720p_2",
  },
  {
    label: "1080p_1",
    detail: "1920×1080, 15fps, 2080Kbps",
    value: "1080p_1",
  },
  {
    label: "1080p_2",
    detail: "1920×1080, 30fps, 3000Kbps",
    value: "1080p_2",
  },
];
var curVideoProfile;
AgoraRTC.onAutoplayFailed = () => {
  alert("click to start autoplay!");
};
AgoraRTC.onMicrophoneChanged = async (changedDevice) => {
  // When plugging in a device, switch to a device that is newly plugged in.
  if (changedDevice.state === "ACTIVE") {
    localTracks.audioTrack.setDevice(changedDevice.device.deviceId);
    // Switch to an existing device when the current device is unplugged.
  } else if (
    changedDevice.device.label === localTracks.audioTrack.getTrackLabel()
  ) {
    const oldMicrophones = await AgoraRTC.getMicrophones();
    oldMicrophones[0] &&
      localTracks.audioTrack.setDevice(oldMicrophones[0].deviceId);
  }
};
AgoraRTC.onCameraChanged = async (changedDevice) => {
  // When plugging in a device, switch to a device that is newly plugged in.
  if (changedDevice.state === "ACTIVE") {
    localTracks.videoTrack.setDevice(changedDevice.device.deviceId);
    // Switch to an existing device when the current device is unplugged.
  } else if (
    changedDevice.device.label === localTracks.videoTrack.getTrackLabel()
  ) {
    const oldCameras = await AgoraRTC.getCameras();
    oldCameras[0] && localTracks.videoTrack.setDevice(oldCameras[0].deviceId);
  }
};
async function initDevices() {
  mics = await AgoraRTC.getMicrophones();
  const audioTrackLabel = localTracks.audioTrack.getTrackLabel();
  currentMic = mics.find((item) => item.label === audioTrackLabel);
  $(".mic-input").val(currentMic.label);
  $(".mic-list").empty();
  mics.forEach((mic) => {
    $(".mic-list").append(`<a class="dropdown-item" href="#">${mic.label}</a>`);
  });

  cams = await AgoraRTC.getCameras();
  const videoTrackLabel = localTracks.videoTrack.getTrackLabel();
  currentCam = cams.find((item) => item.label === videoTrackLabel);
  $(".cam-input").val(currentCam.label);
  $(".cam-list").empty();
  cams.forEach((cam) => {
    $(".cam-list").append(`<a class="dropdown-item" href="#">${cam.label}</a>`);
  });
}
async function switchCamera(label) {
  currentCam = cams.find((cam) => cam.label === label);
  $(".cam-input").val(currentCam.label);
  // switch device of local video track.
  await localTracks.videoTrack.setDevice(currentCam.deviceId);
}
async function switchMicrophone(label) {
  currentMic = mics.find((mic) => mic.label === label);
  $(".mic-input").val(currentMic.label);
  // switch device of local audio track.
  await localTracks.audioTrack.setDevice(currentMic.deviceId);
}
function initVideoProfiles() {
  videoProfiles.forEach((profile) => {
    $(".profile-list").append(
      `<a class="dropdown-item" label="${profile.label}" href="#">${profile.label}: ${profile.detail}</a>`
    );
  });
  curVideoProfile = videoProfiles.find((item) => item.label == "480p_1");
  $(".profile-input").val(`${curVideoProfile.detail}`);
}
async function changeVideoProfile(label) {
  curVideoProfile = videoProfiles.find((profile) => profile.label === label);
  $(".profile-input").val(`${curVideoProfile.detail}`);
  // change the local video track`s encoder configuration
  localTracks.videoTrack &&
    (await localTracks.videoTrack.setEncoderConfiguration(
      curVideoProfile.value
    ));
}

/*
 * When this page is called with parameters in the URL, this procedure
 * attempts to join a Video Call channel using those parameters.
 */
$(() => {
  initVideoProfiles();
  $(".profile-list").delegate("a", "click", function (e) {
    changeVideoProfile(this.getAttribute("label"));
  });
  var urlParams = new URL(location.href).searchParams;
  options.appid = urlParams.get("appid");
  options.channel = urlParams.get("channel");
  options.token = urlParams.get("token");
  options.uid = urlParams.get("uid");
  if (options.appid && options.channel) {
    $("#uid").val(options.uid);
    $("#appid").val(options.appid);
    $("#token").val(options.token);
    $("#channel").val(options.channel);
    $("#join-form").submit();
  }
});

/*
 * When a user clicks Join or Leave in the HTML form, this procedure gathers the information
 * entered in the form and calls join asynchronously. The UI is updated to match the options entered
 * by the user.
 */
$("#join-form").submit(async function (e) {
  e.preventDefault();
  $("#join").attr("disabled", true);
  try {
    client = AgoraRTC.createClient({
      mode: "rtc",
      codec: getCodec(),
    });
    options.channel = $("#channel").val();
    options.uid = Number($("#uid").val());
    options.appid = $("#appid").val();
    options.token = $("#token").val();
    await join();
    if (options.token) {
      $("#success-alert-with-token").css("display", "block");
    } else {
      $("#success-alert a").attr(
        "href",
        `index.html?appid=${options.appid}&channel=${options.channel}&token=${options.token}`
      );
      $("#success-alert").css("display", "block");
    }
  } catch (error) {
    /* suppressed debug */
  } finally {
    $("#leave").attr("disabled", false);
  }
});

/*
 * Called when a user clicks Leave in order to exit a channel.
 */
$("#leave").click(function (e) {
  leave();
});
$("#agora-collapse").on("show.bs.collapse	", function () {
  initDevices();
});
$(".cam-list").delegate("a", "click", function (e) {
  switchCamera(this.text);
});
$(".mic-list").delegate("a", "click", function (e) {
  switchMicrophone(this.text);
});

/*
 * Join a channel, then create local video and audio tracks and publish them to the channel.
 */
async function join() {
  // Add an event listener to play remote tracks when remote user publishes.
  client.on("user-published", handleUserPublished);
  client.on("user-unpublished", handleUserUnpublished);
  options.uid = await client.join(
    options.appid,
    options.channel,
    options.token || null,
    options.uid || null
  );
  $("#captured-frames").css("display", DEBUG_MODE ? "block" : "none");
}

/*
 * Stop all local and remote tracks then leave the channel.
 */
async function leave() {
  for (trackName in localTracks) {
    var track = localTracks[trackName];
    if (track) {
      track.stop();
      track.close();
      localTracks[trackName] = undefined;
    }
  }

  // Remove remote users and player views.
  remoteUsers = {};
  $("#remote-playerlist").html("");

  // leave the channel
  await client.leave();
  $("#local-player-name").text("");
  $("#join").attr("disabled", false);
  $("#leave").attr("disabled", true);
  $("#joined-setup").css("display", "none");
  /* suppressed debug */
}

/*
 * Add the local use to a remote channel.
 *
 * @param  {IAgoraRTCRemoteUser} user - The {@link  https://docs.agora.io/en/Voice/API%20Reference/web_ng/interfaces/iagorartcremoteuser.html| remote user} to add.
 * @param {trackMediaType - The {@link https://docs.agora.io/en/Voice/API%20Reference/web_ng/interfaces/itrack.html#trackmediatype | media type} to add.
 */
async function subscribe(user, mediaType) {
  const uid = user.uid;
  const subscriptionKey = `${uid}:${mediaType}`;

  // Backoff if we've seen an incompatible SDP recently
  const incompatibleSince = incompatibleSubscriptions.get(subscriptionKey);
  if (incompatibleSince && Date.now() - incompatibleSince < 60_000) {

  }

  if (pendingSubscriptions.has(subscriptionKey)) {
    return;
  }
  pendingSubscriptions.add(subscriptionKey);

  try {
    await client.subscribe(user, mediaType);

    if (mediaType === "video") {
      const playerWidth = uid === 1001 ? "540px" : uid === 1000 ? "1024px" : "auto";
      const playerHeight = uid === 1001 ? "360px" : uid === 1000 ? "576px" : "auto";

      if ($(`#player-wrapper-${uid}`).length === 0) {
        const player = $(`
          <div id="player-wrapper-${uid}">
            <p class="player-name">(${uid})</p>
            <div id="player-${uid}" class="player" style="width: ${playerWidth}; height: ${playerHeight};"></div>
          </div>
        `);
        $("#remote-playerlist").append(player);
      }

      try {
        if (user.videoTrack) {
          user.videoTrack.play(`player-${uid}`);
          user.videoTrack.captureEnabled = true;
        } else {
          /* suppressed debug */
        }
      } catch (err) {
        /* suppressed debug */
      }

      if ($(`#captured-frame-${uid}`).length === 0) {
        const capturedFrameDiv = $(`
          <div id="captured-frame-${uid}" style="width: ${playerWidth}; height: ${playerHeight}; display: ${
          DEBUG_MODE ? "block" : "none"
        };">
            <p>Captured Frames (${uid})</p>
            <img id="captured-image-${uid}" style="width: 100%; height: 100%; object-fit: contain;">
            <button id="download-frame-${uid}" class="btn btn-primary mt-2">Download Frame</button>
            <button id="download-base64-${uid}" class="btn btn-secondary mt-2 ml-2">Download Base64</button>
          </div>
        `);
        $("#captured-frames").append(capturedFrameDiv);
      }
    }

    if (mediaType === "audio") {
      if (user.audioTrack) {
        user.audioTrack.play();
      }
    }
  } catch (err) {
    const msg = (err && (err.message || JSON.stringify(err))) || String(err);

    // If it's an SDP send/recv mismatch, avoid repeated retries and notify the bot
    if (msg.includes('Incompatible send direction') || msg.includes('set remote answer error')) {
      incompatibleSubscriptions.set(subscriptionKey, Date.now());
      // schedule removal after 60s so we can try again later
      setTimeout(() => incompatibleSubscriptions.delete(subscriptionKey), 60_000);
      // Send a message to the bot to ask it to republish (if supported)
      try {
        if (window && window.sendMessage) {
          window.sendMessage({ type: 'republish_request', uid: uid, mediaType: mediaType });
        }
      } catch (e) {
        /* suppressed debug */
      }
    }

    // If it's a repeat subscribe, ignore further retries for a short time
    if (msg.includes('Repeat subscribe') || (err && err.code === 2021)) {
    }
  } finally {
    pendingSubscriptions.delete(subscriptionKey);
  }
}

/*
 * Add a user who has subscribed to the live channel to the local interface.
 *
 * @param  {IAgoraRTCRemoteUser} user - The {@link  https://docs.agora.io/en/Voice/API%20Reference/web_ng/interfaces/iagorartcremoteuser.html| remote user} to add.
 * @param {trackMediaType - The {@link https://docs.agora.io/en/Voice/API%20Reference/web_ng/interfaces/itrack.html#trackmediatype | media type} to add.
 */
function handleUserPublished(user, mediaType) {
  const id = user.uid;
  remoteUsers[id] = user;
  subscribe(user, mediaType);
}

/*
 * Remove the user specified from the channel in the local interface.
 *
 * @param  {string} user - The {@link  https://docs.agora.io/en/Voice/API%20Reference/web_ng/interfaces/iagorartcremoteuser.html| remote user} to remove.
 */
function handleUserUnpublished(user, mediaType) {
  if (mediaType === "video") {
    const id = user.uid;
    delete remoteUsers[id];
    $(`#player-wrapper-${id}`).remove();
  }
}
function getCodec() {
  var radios = document.getElementsByName("radios");
  var value;
  for (var i = 0; i < radios.length; i++) {
    if (radios[i].checked) {
      value = radios[i].value;
    }
  }
  return value;
}

async function captureFrameAsBase64(videoTrack) {
  if (!videoTrack) {
    return null;
  }
  
  // First try using getCurrentFrameData (Agora SDK method)
  if (typeof videoTrack.getCurrentFrameData === 'function') {
    try {
      const frame = await videoTrack.getCurrentFrameData();
      if (frame && frame.width && frame.height) {
        const canvas = document.createElement("canvas");
        canvas.width = frame.width;
        canvas.height = frame.height;
        const ctx = canvas.getContext("2d");
        ctx.putImageData(frame, 0, 0);
        return canvas.toDataURL(
          `image/${window.imageParams["imageFormat"]}`,
          window.imageParams["imageQuality"]
        );
      }
    } catch (err) {
      /* suppressed debug */
    }
  }
  
  // Fallback: Find the video element in the DOM and capture from it
  try {
    // Try to get the video element from the track's internal player
    const trackId = videoTrack.getTrackId ? videoTrack.getTrackId() : null;
    let videoEl = null;
    
    // Look for video element associated with this track
    if (trackId) {
      videoEl = document.querySelector(`video[id*="${trackId}"]`);
    }
    
    // If not found by track ID, try to find by player
    if (!videoEl) {
      // The Agora SDK creates video elements inside player divs
      const allVideos = document.querySelectorAll('.player video, .agora_video_player');
      for (const v of allVideos) {
        if (v.readyState >= 2 && v.videoWidth > 0) {
          videoEl = v;
          break;
        }
      }
    }
    
    if (videoEl && videoEl.readyState >= 2 && videoEl.videoWidth > 0) {
      const canvas = document.createElement("canvas");
      canvas.width = videoEl.videoWidth;
      canvas.height = videoEl.videoHeight;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(videoEl, 0, 0);
      return canvas.toDataURL(
        `image/${window.imageParams["imageFormat"]}`,
        window.imageParams["imageQuality"]
      );
    } else {
      return null;
    }
  } catch (err) {
    return null;
  }
}

// Add at the beginning of the file
const DEBUG_MODE = false;
const lastBase64Frames = {};

// Function to get the latest base64 frame for a specific UID
async function getLastBase64Frame(uid) {
  try {
    const user = remoteUsers[uid];
    
    // First try using the user's videoTrack
    if (user && user.videoTrack) {
      const base64Frame = await captureFrameAsBase64(user.videoTrack);
      if (base64Frame) {
        lastBase64Frames[uid] = base64Frame;
        return base64Frame;
      }
    }
    
    // Fallback: Try to find a video element for this UID in the DOM
    const playerDiv = document.querySelector(`#player-${uid}`);
    if (playerDiv) {
      const videoEl = playerDiv.querySelector('video');
      if (videoEl && videoEl.readyState >= 2 && videoEl.videoWidth > 0) {
        const canvas = document.createElement("canvas");
        canvas.width = videoEl.videoWidth;
        canvas.height = videoEl.videoHeight;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(videoEl, 0, 0);
        const base64Frame = canvas.toDataURL(
          `image/${window.imageParams["imageFormat"]}`,
          window.imageParams["imageQuality"]
        );
        lastBase64Frames[uid] = base64Frame;
        return base64Frame;
      } else {
      }
    } else {
    }
    
    return null;
  } catch (err) {
    return null;
  }
}

function initializeImageParams({ imageFormat, imageQuality }) {
  window.imageParams = { imageFormat, imageQuality };
}
window.initializeImageParams = initializeImageParams;
window.getLastBase64Frame = getLastBase64Frame