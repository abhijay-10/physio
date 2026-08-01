export interface ModuleInfo {
  name: string;
  folder: string;
  icon: string;
  description?: string;
  instructions?: string[];
  benefits?: string;
  hologramUrl?: string;
  analysisMetrics?: {
    targetAngle?: string;
    keyPoints?: number;
    focusAreas?: string[];
  };
}

export const CHEST_MODULES: ModuleInfo[] = [
  { 
    name: "Back Pose", 
    folder: "chest/back_pose", 
    icon: "🧘",
    description: "Evaluates the posterior chest wall, thoracic spine alignment, and shoulder symmetry from the rear view.",
    instructions: [
      "Stand tall facing away from the scanner.",
      "Align the heels and spine centered with the scanning line.",
      "Keep the arms relaxed at the sides with shoulders level.",
      "Maintain a natural, erect posture."
    ],
    benefits: "Provides baseline measurements of posterior spinal curves, scapular height differences, and back muscle balance.",
    hologramUrl: "/holograms/hologram_chest.png",
    analysisMetrics: {
      targetAngle: "0° (Neutral Standing)",
      keyPoints: 10,
      focusAreas: ["Scapular Symmetry", "Thoracic Spine Alignment", "Shoulder Line"]
    }
  },
  { 
    name: "Front Pose", 
    folder: "chest/lordotic_front_pose", 
    icon: "👤",
    description: "Captures the anterior chest profile to check lordotic alignment, clavicle leveling, and posture balance.",
    instructions: [
      "Stand facing the scanner with feet shoulder-width apart.",
      "Align the chest and sternum centered with the scanner axis.",
      "Keep shoulders back and chin level.",
      "Breathe normally during the quick scan sequence."
    ],
    benefits: "Aids in assessing chest wall deformities like pectus excavatum, clavicular symmetry, and forward head posture.",
    hologramUrl: "/holograms/hologram_chest.png",
    analysisMetrics: {
      targetAngle: "0° (Anterior Front)",
      keyPoints: 10,
      focusAreas: ["Clavicle Levels", "Sternal Alignment", "Chest Expansion"]
    }
  },
  { 
    name: "Sleep Front", 
    folder: "chest/sleep_front", 
    icon: "🛌",
    description: "Monitors sleep posture in the prone position (sleeping face-down) to analyze mechanical strain on the ribcage and spine.",
    instructions: [
      "Lie face-down on the scanning surface.",
      "Turn the head comfortably to one side.",
      "Keep the arms relaxed at the sides or raised upward.",
      "Avoid arching the lower back excessively."
    ],
    benefits: "Helps identify pressure points on the chest, ribcage constriction, and neck rotation strain during prone sleep.",
    hologramUrl: "/holograms/hologram_chest.png",
    analysisMetrics: {
      targetAngle: "180° Flat",
      keyPoints: 12,
      focusAreas: ["Ribcage Pressure", "Cervical Rotation", "Spine Support"]
    }
  },
  { 
    name: "Back Front", 
    folder: "chest/sleep_back", 
    icon: "🔄",
    description: "Analyzes supine sleep posture (sleeping face-up) to examine alignment of the chest, thoracic spine, and shoulder support.",
    instructions: [
      "Lie flat on your back on the scanning platform.",
      "Keep the legs straight and arms resting naturally at your sides.",
      "Ensure the spine is fully supported by the surface.",
      "Look directly upward toward the ceiling."
    ],
    benefits: "Assesses spine alignment, scapular contact areas, and breathing patterns when sleeping on the back.",
    hologramUrl: "/holograms/hologram_chest.png",
    analysisMetrics: {
      targetAngle: "180° Flat",
      keyPoints: 12,
      focusAreas: ["Scapular Contact", "Thoracic Curve", "Sternum Horizon"]
    }
  },
  { 
    name: "Sitting Front", 
    folder: "chest/sitting_front_pose", 
    icon: "🪑",
    description: "Evaluates sitting posture from the front to evaluate ergonomic alignment, slouching, and weight distribution on the torso.",
    instructions: [
      "Sit on the stool facing the scanner with feet flat on the floor.",
      "Maintain your typical sitting posture (do not artificially over-correct).",
      "Rest hands on thighs, keeping the forearms relaxed.",
      "Keep head facing straight forward."
    ],
    benefits: "Essential for ergonomic assessments, finding seated shoulder tilt, and identifying torso slouch patterns.",
    hologramUrl: "/holograms/hologram_chest.png",
    analysisMetrics: {
      targetAngle: "90° (Hip-Thigh Angle)",
      keyPoints: 10,
      focusAreas: ["Shoulder Leveling", "Torso Tilt", "Head Position"]
    }
  }
];

export const HAND_MODULES: ModuleInfo[] = [
  { 
    name: "Bilateral Hand", 
    folder: "hand/bilateralhand", 
    icon: "👐",
    description: "The Bilateral Hand scan simultaneously evaluates both hands, often used for comparative analysis in systemic conditions like rheumatoid arthritis.",
    instructions: [
      "Place both hands flat against the detector surface side-by-side.",
      "Ensure fingers are spread equally on both hands.",
      "Keep the wrists and forearms aligned neutrally.",
      "Hold completely still during the dual capture process."
    ],
    benefits: "Provides an immediate symmetrical comparison, highlighting unilateral joint degradation or swelling versus bilateral systemic changes.",
    hologramUrl: "/holograms/hologram_bilateral.png",
    analysisMetrics: {
      targetAngle: "0° (Flat)",
      keyPoints: 42,
      focusAreas: ["Symmetry", "Joint Space", "Phalangeal Alignment"]
    }
  },
  { 
    name: "Fan Lateral", 
    folder: "hand/fanlateral", 
    icon: "🖐️",
    description: "The Fan Lateral view spaces out the digits to avoid superimposition, allowing individual assessment of the phalanges and metacarpals in a lateral profile.",
    instructions: [
      "Rest the ulnar side (pinky side) of the hand on the surface.",
      "Fan out the fingers so they resemble a staircase or fan.",
      "Ensure each finger is clearly separated from the others.",
      "Keep the thumb extended and parallel to the detector."
    ],
    benefits: "Crucial for identifying subtle fractures in individual digits that would otherwise be obscured by overlapping bones in a standard lateral view.",
    hologramUrl: "/holograms/hologram_fan_lateral.png",
    analysisMetrics: {
      targetAngle: "15-20° per digit separation",
      keyPoints: 21,
      focusAreas: ["Digit Separation", "Metacarpal Profile", "Phalangeal Shafts"]
    }
  },
  { 
    name: "Lateral Hand", 
    folder: "hand/lateralhand", 
    icon: "✋",
    description: "The True Lateral Hand view is primarily used to assess anterior or posterior displacement of fractures and for locating foreign bodies within the soft tissue.",
    instructions: [
      "Rest the ulnar side (pinky side) of the hand completely flat on the detector.",
      "Extend all fingers and perfectly superimpose them over one another.",
      "Extend the thumb outward, parallel to the detector surface.",
      "Ensure the hand remains perpendicular to the surface without rotating."
    ],
    benefits: "Offers an unobstructed view of the metacarpal displacement and helps in confirming joint subluxations and foreign body depths.",
    hologramUrl: "/holograms/hologram_lateral.png",
    analysisMetrics: {
      targetAngle: "90° (Perpendicular)",
      keyPoints: 21,
      focusAreas: ["Superimposition", "Anterior/Posterior Displacement", "Foreign Body Depth"]
    }
  },
  { 
    name: "Oblique Hand", 
    folder: "hand/obliquehand", 
    icon: "🖖",
    description: "The Oblique Hand view provides a perspective angled halfway between PA and Lateral, isolating the metacarpals and phalanges to expose hidden fractures.",
    instructions: [
      "Start with the hand resting flat on the palmar surface.",
      "Rotate the hand outward (laterally) about 45 degrees.",
      "Use a positioning wedge if available, resting the fingers parallel to the film.",
      "Keep the digits slightly separated to prevent overlapping."
    ],
    benefits: "Excellent for visualizing the metacarpal heads and shafts without superimposition, making it standard for suspected hand trauma.",
    hologramUrl: "/holograms/hologram_oblique.png",
    analysisMetrics: {
      targetAngle: "45° Oblique",
      keyPoints: 21,
      focusAreas: ["Metacarpal Heads", "Phalangeal Shafts", "Joint Clearances"]
    }
  },
  { 
    name: "PA Hand", 
    folder: "hand/pa_hand", 
    icon: "🤚",
    description: "The Posteroanterior (PA) Hand view is a standard projection used to evaluate the carpal bones, metacarpals, and phalanges for fractures, dislocations, or arthritic changes.",
    instructions: [
      "Rest the palmar surface of the hand flat against the detector.",
      "Spread the fingers slightly apart to prevent overlap.",
      "Ensure the wrist is in a neutral, relaxed position.",
      "Keep the hand completely still during the scan."
    ],
    benefits: "Provides a clear overview of the entire hand anatomy, essential for initial trauma assessment and joint space evaluation.",
    hologramUrl: "/holograms/hologram_pa_hand.png",
    analysisMetrics: {
      targetAngle: "0° (Flat)",
      keyPoints: 21,
      focusAreas: ["Carpal Bones", "Metacarpals", "Phalanges Overview"]
    }
  },
  { 
    name: "PA 3-Finger", 
    folder: "hand/pa3finger", 
    icon: "✌️",
    description: "A focused Posteroanterior (PA) view isolating the three central digits (Index, Middle, Ring), often used when specific trauma involves these specific fingers.",
    instructions: [
      "Place the hand flat against the detector with the palm down.",
      "Isolate and extend the index, middle, and ring fingers.",
      "Keep the thumb and pinky finger relaxed and out of the primary scan area if instructed.",
      "Hold the designated fingers straight and still."
    ],
    benefits: "Reduces radiation and focuses the AI analysis strictly on the central digits for highly detailed structural assessment.",
    hologramUrl: "/holograms/hologram_pa3finger.png",
    analysisMetrics: {
      targetAngle: "0° (Flat)",
      keyPoints: 12,
      focusAreas: ["Index Finger", "Middle Finger", "Ring Finger"]
    }
  },
  { 
    name: "Oblique Thumb", 
    folder: "hand/obliquethumb", 
    icon: "👍",
    description: "The Oblique Thumb scan specifically targets the first digit, highlighting the interphalangeal and metacarpophalangeal joints of the thumb.",
    instructions: [
      "Rest the palm flat against the detector surface.",
      "The thumb naturally falls into a 45-degree oblique position in this posture.",
      "Ensure the thumb is extended and separated from the index finger.",
      "Keep the hand relaxed but steady."
    ],
    benefits: "Provides the best visualization of the thumb's bony structures, particularly useful for diagnosing Bennett's or Rolando's fractures.",
    hologramUrl: "/holograms/hologram_oblique_thumb.png",
    analysisMetrics: {
      targetAngle: "45° (Thumb Only)",
      keyPoints: 4,
      focusAreas: ["Interphalangeal Joint", "Metacarpophalangeal Joint", "First Metacarpal"]
    }
  }
];

export const SPINE_MODULES: ModuleInfo[] = [
  { 
    name: "Lateral Spine Scan", 
    folder: "spine", 
    icon: "🦴",
    description: "The Lateral Spine scan evaluates the alignment of the cervical, thoracic, and lumbar vertebrae from a side profile to detect curvature abnormalities or disc compression.",
    instructions: [
      "Stand perpendicular to the scanner with arms raised forward or folded.",
      "Align the spine vertically with the sagittal positioning guide.",
      "Keep the shoulders relaxed and avoid torso rotation.",
      "Hold breath and remain perfectly still during the exposure."
    ],
    benefits: "Essential for measuring lordosis, kyphosis, vertebral heights, and checking for spinal subluxations.",
    hologramUrl: "/holograms/hologram_spine.png",
    analysisMetrics: {
      targetAngle: "90° (True Lateral)",
      keyPoints: 17,
      focusAreas: ["Vertebral Alignment", "Disc Space Height", "Spinal Curvature"]
    }
  }
];

export const ELBOW_MODULES: ModuleInfo[] = [
  { 
    name: "Straight Desk Baseline", 
    folder: "elbow/straight", 
    icon: "📏",
    description: "The Straight Desk Baseline scan assesses the elbow in a fully extended position resting on a flat surface, capturing the carrying angle of the joint.",
    instructions: [
      "Extend the arm fully on the desk surface with the palm facing up.",
      "Ensure the shoulder and elbow joint are at the same horizontal level.",
      "Align the humeral epicondyles parallel to the surface.",
      "Keep fingers relaxed but fully extended."
    ],
    benefits: "Allows measurement of the carrying angle and reveals joint space narrowing in full extension.",
    hologramUrl: "/holograms/hologram_elbow_straight.png",
    analysisMetrics: {
      targetAngle: "180° (Full Extension)",
      keyPoints: 7,
      focusAreas: ["Carrying Angle", "Humero-ulnar Joint", "Epicondylar Alignment"]
    }
  },
  { 
    name: "Lateral 90°", 
    folder: "elbow/elbow90", 
    icon: "💪",
    description: "The Lateral 90° elbow scan is a standard diagnostic view of the elbow joint bent at a right angle, highlighting the olecranon process and soft tissue fat pads.",
    instructions: [
      "Flex the elbow to a precise 90-degree angle.",
      "Place the ulnar side of the hand and wrist flat on the table.",
      "Position the shoulder at the same height as the elbow joint.",
      "Keep the forearm perpendicular to the upper arm."
    ],
    benefits: "Exposes displacement of the posterior fat pad (sail sign), indicating joint effusion or occult radial head fractures.",
    hologramUrl: "/holograms/hologram_elbow_90.png",
    analysisMetrics: {
      targetAngle: "90° (Flexion)",
      keyPoints: 7,
      focusAreas: ["Olecranon Process", "Radial Head", "Fat Pad Displacement"]
    }
  },
  { 
    name: "Humerus AP Partial", 
    folder: "elbow/humerus", 
    icon: "📐",
    description: "The AP Humerus Partial Flexion scan is utilized when a patient cannot fully extend their elbow, allowing evaluation of the distal humerus.",
    instructions: [
      "Rest the upper arm (humerus) flat against the surface.",
      "Support the forearm in a partially flexed posture using a wedge if needed.",
      "Ensure the posterior humerus remains parallel to the detector.",
      "Keep the shoulder aligned and still."
    ],
    benefits: "Enables clear visualization of the distal humerus bone structure and epicondyles despite limited mobility.",
    hologramUrl: "/holograms/hologram_elbow_humerus.png",
    analysisMetrics: {
      targetAngle: "120-140° Partial Flexion",
      keyPoints: 7,
      focusAreas: ["Distal Humerus", "Medial/Lateral Epicondyles", "Trochlea"]
    }
  },
  { 
    name: "Humerus Jones AP Acute", 
    folder: "elbow/humerusjones", 
    icon: "💥",
    description: "The Jones Method (Acute Flexion AP) scan evaluates the distal humerus and proximal forearm structures when the joint is fully flexed.",
    instructions: [
      "Flex the elbow completely so the hand rests on the shoulder if possible.",
      "Position the posterior surface of the flexed elbow flat on the detector.",
      "Ensure the forearm is superimposed directly over the humerus.",
      "Maintain stability throughout the scan."
    ],
    benefits: "Allows critical assessment of the olecranon process and distal humerus outline in cases of severe trauma or contracture.",
    hologramUrl: "/holograms/hologram_elbow_jones.png",
    analysisMetrics: {
      targetAngle: "30-45° Acute Flexion",
      keyPoints: 7,
      focusAreas: ["Olecranon Contour", "Radial Head Overlap", "Distal Humeral Shaft"]
    }
  },
  { 
    name: "PA Axial Olecranon", 
    folder: "elbow/olecaran", 
    icon: "🦾",
    description: "The PA Axial Olecranon scan provides an specialized view of the olecranon fossa and olecranon process, highlighting subtle bone fragments or osteophytes.",
    instructions: [
      "Place the flexed elbow on the table with the forearm upright.",
      "Angle the upper arm forward slightly to project the olecranon process.",
      "Align the sensor beam directly through the olecranon fossa.",
      "Avoid lateral tilt of the forearm."
    ],
    benefits: "Highly effective at diagnosing posterior elbow impingement, olecranon bone spurs, and intra-articular loose bodies.",
    hologramUrl: "/holograms/hologram_elbow_olecranon.png",
    analysisMetrics: {
      targetAngle: "110° Angle",
      keyPoints: 7,
      focusAreas: ["Olecranon Fossa", "Olecranon Tip", "Ulnar Nerve Groove"]
    }
  }
];

export const KNEE_MODULES: ModuleInfo[] = [
  { 
    name: "Hungsten", 
    folder: "knee/hungsten", 
    icon: "🦵",
    description: "The Hungsten view evaluates the patellofemoral joint space and patellar alignment from an axial tangential angle.",
    instructions: [
      "Lie prone on the examination table.",
      "Flex the knee to a 45-degree angle.",
      "Use the patient strap or support to keep the lower leg steady.",
      "Ensure the patella is centered and aligned with the detector."
    ],
    benefits: "Indispensable for diagnosing patellar subluxation, tracking syndromes, and osteochondral fractures of the patella.",
    hologramUrl: "/holograms/hologram_knee.png",
    analysisMetrics: {
      targetAngle: "45° Flexion",
      keyPoints: 8,
      focusAreas: ["Patellofemoral Joint", "Patella Tracking", "Femoral Condyles"]
    }
  },
  { 
    name: "PA Knee", 
    folder: "knee/pa_knee", 
    icon: "🦿",
    description: "The Posteroanterior (PA) weight-bearing knee scan assesses the tibiofemoral joint space under physiological loading conditions.",
    instructions: [
      "Stand upright with the front of the knees facing the detector.",
      "Distribute weight evenly on both feet.",
      "Flex the knees slightly (about 10-15 degrees) if performing a flexion view.",
      "Remain perfectly still during the weight-bearing capture."
    ],
    benefits: "Reveals cartilage loss and joint space narrowing much more accurately than non-weight-bearing supine scans.",
    hologramUrl: "/holograms/hologram_knee.png",
    analysisMetrics: {
      targetAngle: "0-15° Flexion",
      keyPoints: 8,
      focusAreas: ["Tibiofemoral Space", "Intercondylar Eminence", "Joint Symmetry"]
    }
  },
  { 
    name: "Patella Lateral", 
    folder: "knee/patella_lateral", 
    icon: "🦵",
    description: "The Patella Lateral view evaluates the patellofemoral joint space and patellar alignment.",
    instructions: [
      "Lie on the examination table or bed.",
      "Flex the knee to a 75 to 115-degree angle.",
      "Ensure the knee is raised higher than the ankle.",
      "Hold perfectly still during capture."
    ],
    benefits: "Essential for evaluating lateral patellar displacement and joint abnormalities.",
    hologramUrl: "/holograms/hologram_patella_lateral.png",
    analysisMetrics: {
      targetAngle: "75-115° Flexion",
      keyPoints: 8,
      focusAreas: ["Patellofemoral Joint", "Patella Tracking"]
    }
  }
];

export const FOOT_MODULES: ModuleInfo[] = [
  { 
    name: "Flat Leg Posture", 
    folder: "foot/flat_leg", 
    icon: "🦿",
    description: "The Flat Leg scan isolates the leg resting flat on the surface while the other leg is bent, capturing the alignment of the extended leg automatically.",
    instructions: [
      "Lie on your back on the examination table.",
      "Extend the target leg completely flat on the detector.",
      "Bend the other knee out of the way.",
      "Hold completely still during the capture."
    ],
    benefits: "Ideal for isolating one leg for precise AP measurements automatically detecting which leg is straight.",
    hologramUrl: "/holograms/hologram_leg.png",
    analysisMetrics: {
      targetAngle: "180° (Straight Flat Leg)",
      keyPoints: 10,
      focusAreas: ["Extended Knee Alignment", "Isolated Leg Horizon"]
    }
  },
  { 
    name: "Back Leg Posture", 
    folder: "foot/foot_angle", 
    icon: "🦶",
    description: "The Back Leg Posture scan measures the vertical alignment and straightness of the legs from the back view, assessing joint symmetry and posture.",
    instructions: [
      "Stand straight facing away from the camera.",
      "Keep both legs straight and fully extended.",
      "Ensure both knee joints and ankle joints are visible.",
      "Hold completely still during the capture."
    ],
    benefits: "Essential for assessing sagittal alignment, pelvic tilt, and knee straightness from the posterior view.",
    hologramUrl: "/holograms/hologram_back_leg.png",
    analysisMetrics: {
      targetAngle: "180° (Straight Leg)",
      keyPoints: 10,
      focusAreas: ["Knee Joint Alignment", "Pelvic Tilt Horizon", "Ankle Joint Horizon"]
    }
  },
  { 
    name: "Front Leg Posture", 
    folder: "foot/foot_ap", 
    icon: "🦵",
    description: "The Anteroposterior (AP) Leg scan captures the alignment of the tibia and fibula, assessing joint margins from the knee down to the ankle from the front view.",
    instructions: [
      "Position the leg centered with the scanning line facing towards the camera.",
      "Keep the leg straight and fully extended.",
      "Ensure the knee joint and ankle joint are visible.",
      "Hold completely still during the exposure."
    ],
    benefits: "Essential for assessing leg length discrepancies, mechanical axis alignment, and tibia/fibula fractures.",
    hologramUrl: "/holograms/hologram_leg.png",
    analysisMetrics: {
      targetAngle: "180° (Straight Leg)",
      keyPoints: 10,
      focusAreas: ["Knee Joint Alignment", "Tibia-Fibula Axis", "Ankle Joint Horizon"]
    }
  },
  { 
    name: "Lateral view of entire tibia and fibula", 
    folder: "foot/lateral_tibia_fibula", 
    icon: "🦵",
    description: "Lateral view of entire tibia and fibula. Central ray Perpendicular to IR at midpoint of shin.",
    instructions: [
      "Patient toward affected side with leg on IR.",
      "Adjust body's rotation to place patella perpendicular to IR.",
      "Use supports where needed for patient's comfort and to maintain body position.",
      "Lift leg enough for assistant to slide rigid support under patient's leg."
    ],
    benefits: "Include proximal and distal ends of tibia and fibula. If patient must remain supine, the image may be taken cross-table using horizontal CR.",
    hologramUrl: "/holograms/hologram_lateral_tibia_fibula.png",
    analysisMetrics: {
      targetAngle: "90° (Lateral View)",
      keyPoints: 8,
      focusAreas: ["Tibia", "Fibula", "Patella"]
    }
  }
];

export const LOWERBACK_MODULES: ModuleInfo[] = [
  { 
    name: "Front AP", 
    folder: "lowerback/front_ap", 
    icon: "🛏️",
    description: "The AP Lumbar Spine view evaluates the alignment of the lower back in a supine position with knees flexed.",
    instructions: [
      "Lie completely flat on your back.",
      "Bend your knees to approximately 120 degrees.",
      "Keep the torso horizontal on the surface.",
      "Remain still during the capture."
    ],
    benefits: "Essential for flattening the lordotic curve to evaluate lumbar vertebrae.",
    hologramUrl: "/holograms/hologram_lower_back.png",
    analysisMetrics: {
      targetAngle: "120° Knee Flexion",
      keyPoints: 10,
      focusAreas: ["Lumbar Spine", "Knee Flexion", "Torso Alignment"]
    }
  }
];

export const ARM_MODULES: ModuleInfo[] = [
  { 
    name: "Back Arm Trauma Position", 
    folder: "arm/trauma_forearm", 
    icon: "💪",
    description: "Evaluates back arm trauma positioning with forearm and hand straight.",
    instructions: [
      "Place forearm flat on the detector surface.",
      "Keep the hand straight and in line with the forearm.",
      "Ensure the arm is vertically aligned in the camera view."
    ],
    benefits: "Essential for proper forearm X-ray positioning in trauma cases.",
    hologramUrl: "/holograms/hologram_elbow_90.png",
    analysisMetrics: {
      targetAngle: "180° (Straight Forearm)",
      keyPoints: 3,
      focusAreas: ["Wrist Alignment", "Vertical Orientation"]
    }
  }
];

export const CATEGORIES = [
  { id: "dashboard", name: "Dashboard Home", icon: "🏠" },
  { id: "chest", name: "Chest Radiology", icon: "🫁" },
  { id: "hand", name: "Hand Postures", icon: "🖐️" },
  { id: "arm", name: "Arm Postures", icon: "💪" },
  { id: "spine", name: "Spine Vertebrae", icon: "🦴" },
  { id: "elbow", name: "Elbow Joint Profile", icon: "🦾" },
  { id: "knee", name: "Knee Diagnostics", icon: "🦵" },
  { id: "foot", name: "Foot Analytics", icon: "🦶" },
  { id: "lowerback", name: "Lower Back Analysis", icon: "🛏️" }
];
