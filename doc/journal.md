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
