# Journal de bord

### 04/03/2020

Discussion avec M. Formenti qui nous présente la situation de l'entreprise et son problème.

Il s'agit d’une startup qui propose des services d’optimisation de consommation de l’eau dans le milieu agricole. Dans cette optique, cette entreprise a installé des senseurs sur des pompes à eaux qui en indiquant la pression à l'intérieur, permettent d'indiquer lorsque l'irrigation est en cours. L'irrigation est déclenchée automatiquement à l'aide de capteurs de l'humidité du sol. 

Il peut y avoir plusieurs senseurs reliés, par exemple deux senseurs aux deux extremités d'une pompe. Ces senseurs envoient en permanence des statistiques à une salle de contrôle ou un employé de l'entreprise observe ces graphes défiler, et se charge de détecter toute anomalie. La détection d'anomalies est donc entièrement manuelle pour l'instant. Si cet employé observe un problème, il peut décider d'envoyer une personne sur place pour le réparer.

Ces senseurs ne sont pas parfaits, et ils peuvent envoyer des valeurs légèrement différentes entre deux lectures. De plus, certains facteurs comme la chaleur ou l'humidité qui varient naturellement font en sorte que les courbes de ces senseurs ont toujours un certain bruit de basse amplitude. Il peut aussi arriver qu'une bulle d'air ou un petit objet passe temporairement par la pompe et devienne une "dent" sur le graphe, c'est à dire une forte variation de l'amplitude en un court moment.

Le fonctionnement normal d'une pompe ressemble donc à une période calme perturbée uniquement par le bruit, puis lorsque le capteur détecte que l'irrigation est nécessaire la courbe grimpe jusqu'à son maximum, et y reste jusqu'à ce que le capteur détecte que la terre est suffisamment irriguée. S'ensuit une chute brusque de la pression puis un retour au calme jusqu'au prochain cycle.

Plusieurs types d'anomalies dans ce cycle sont possibles:

 - Si deux senseurs sont présents et que la pression de l'un est sensiblement plus basse que celle de l'autre, cela peut être une indication que la pompe est bouchée.
 - Si un senseur subit de grandes variations d'amplitude alors que l'irrigation est en cours.
 - ...

Le but de ce TER est donc d'automatiser la détection de ces anomalies automatiquement à l'aide d'outils d'apprentissage profond.

Afin de détecter ces anomalies, nous voudrions convertir ces graphes en images ou vidéos et appliquer des algorithmes de classification d'image dans lesquels les réseaux de neurones sont particulièrement efficaces. Cela permettrait d'utiliser le même programme pour traiter des données complètement différentes sans devoir le réentraîner pour chaque jeu de données.

La première étape que nous nous sommes fixés est de créer un outil permettant de génerer des graphes ressemblant aux graphes utilisés par l'entreprise. Cet outil devrait être modulaire pour permettre de simuler toutes les situations auxquelles l'entreprise est confrontée, incluant les anomalies. Il faut donc réfléchir à des paramètres permettant de varier le temps entre les "cycles" de pression, le nombre d'anomalies, leur type, le bruit, etc.

### 06/03/2020



Discussion sur la conception du logiciel de simulation.

La question principale: Est-ce que le logiciel génerera des graphes réalistes tout seul pour entraîner le modèle de machine learning, ou bien est-ce que le logiciel prendra en entrée les fichiers .csv fournis à M. Formenti par l'entreprise et qui correspondent aux relevés des senseurs.

En attendant, nous avons réfléchi à la conception du logiciel en partant du premier point. Nous voulons partir sur une architecture à deux temps: un programme qui génère des fichiers .json qui correspondent aux informations d'un graphe, et un autre logiciel qui lit ces .json et les transforme en graphes.

La structure des fichiers .json est la suivante:

		realtime_tick: int
            Ticks in <milliseconds> before registering a sample and adding
            `dt` to the simulation's time.
        dt_per_sample: int
            Simulated time step between each sample in the simulation.
            This variable will likely be in <seconds>.
        transition_type: TransitionType
            The TransitionType between states. (e.g. EaseInOutQuad... etc.)
        noise: Noise
            The general Noise that is present for the entire system's samples.
            This simulates general noise such as Temperature, Humidity  as well
            as other factors that interfer with the sensors' readings.
        states: List[State]
            The list of States that this system emulates.
            Attributes: 
	            state: str [Active / Rest State]
			    duration: int [Duration of state]
			    amplitude: int [Avg. amplitude of state]
			    impulsions: List[Impulsion]
			    	Impulsion attributes:
					    type: str  [ex: "Dent"]
					    fire_time: int [How many seconds into the state is the impulsion supposed to start]
					    duration: int [Duration of impulsion]

Exemple:

	{
		"realtime_tick": 42,
		"dt_per_sample": 60,
		"transition_type": "EaseInOut",
		"noise": "gaussian",
		"states": [
					{
						"state": "active_state",
						"duration": 24,
						"amplitude": 100,
						"impulsions" : [
							{
								"type": "dent",
								"fire_time": 3,
								"duration": 5
							},
							{
								"type": "dent",
								"fire_time": 14,
								"duration": 2
							},
						]
					},
					{
						"state": "rest_state",
						"duration": 24,
						"amplitude": 100,
						"impulsions" : []
					},
				]
	}

Nous avons choisi de représenter les graphes sous la forme d'une suite d'états (actif, puis repos, puis actif...) avec chacun une durée et une amplitude moyenne.
Certains paramètres son globaux et s'appliquent à tous les états d'un graphe: la fonction de bruit (bruit "normal" ou bruit aléatoire par exemple, qui dévie un peu de l'amplitude moyenne), ainsi que le type de transition entre les états (par exemple pour décider à quel point la transition entre les deux états est abrupte). Afin de simuler les anomalies (graves ou non), nous avons implémenté des "impulsions" qui sont des grandes variations de l'amplitude. Chaque type d'impulsion déformera différemment le signal pour simuler une anomalie différente.

L'intérêt de ce système à deux temps est que le programme qui génère les fichiers .json peut lui-même prendre en compte des probabilités pour décider d'à quel moment insérer une anomalie, ou à quel moment arrêter un état. On peut génerer autant de graphes que l'on veut avec le type d'anomalie désiré, et avoir un dataset équilibré.

###20/03/2020

Discussion avec M. Formenti pour constater notre avancement et discuter de la marche à suivre.

Le programme qui interprête le json vers une image de graphe implémente bien les impulsions, les différentes transitions possibles, et les suites d'état. Toute la surface "en dessous" de la ligne est coloriée avec opacité 1 de la même couleur pour faciliter le traitement par les algorithmes de classification d'images.

Afin de pouvoir alimenter notre modèle avec un maximum de graphes (par exemple, un qui représente la situation entre deux anomalies), nous avons implémentés deux nouvelles options : Scale et Offset.

L'offset décide de la "position" de départ dans le temps, le scale décide de combien de secondes afficher à la fois.

Série d'exemples pour illustrer ces deux fonctions:

![Graphe 1](https://i.imgur.com/5zNGcO2.png)

		ci-dessus: offset = 0, scale = 0 (default), noise = 0. Chaque "dent" est une impulsion.

![Graphe 2](https://i.imgur.com/Dvgafj2.png)

		ci-dessus: offset = 0, scale = 2000, noise = 0

![Graphe 3](https://i.imgur.com/NxLtvLR.png)

		ci-dessus: offset = 2000, scale = 0, noise = 0

![Graphe 4](https://i.imgur.com/gafReV3.png)

		ci-dessus: offset = 2000, scale = 2000, noise = 5

Le traitement du ymax/ymin de chaque graphe n'est plus automatisé par matplotlib mais calculé à partir de toutes les données en entrée, sans limiter par scale et offset, pour que le même graphe avec des valeurs différentes de scale et offset aie toujours la même échelle.

Avec cette technique de scale/offset, on peut génerer toutes les images individuelles d'un graphe sous forme de stream comme celui qui est surveillé par l'entreprise, en donnant à chaque fois le même graphe en entrée avec un offset incrémenté.

Comme l'entreprise a besoin que la détection d'anomalie se fasse de manière quasi-instantanée, nous voulons choisir des algorithmes d'apprentissage Deep Learning qui arrivent en fin de compte à discerner des motifs que le cerveau humain arrive à distinguer trivialement.

Les prochains objectifs:
* Réaliser le programme qui génère les json (en se servant de probabilités pour simuler la fréquence des anomalies)
* Générer plusieurs images étiquettées à partir d'un graphe précédemment réalisé qui va nous servir comme données pour entraîner notre modèle.
* Comparer les modèles de Convolutional Neural Networks qui permettent de faire de la classification d'images pour trouver le plus adapté.
